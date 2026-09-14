import os
import threading
import time
import numpy as np
import torch
import queue
import torch.distributed as dist
try:
    import dlimp as dl
except ImportError:
    class DummyDL:
        Dataset = None
    dl = DummyDL()
from typing import Optional, Any, List, Union, Tuple
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from .dataset_wrappers import WrappedDataset, WrappedIterableDataset, MapToIterableDataset

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _collect_cache_shard_ranges(dataset, offset=0):
    """Collect global index ranges without importing the cache module."""
    ranges = getattr(dataset, 'shard_ranges', None)
    if ranges:
        return [(offset + int(start), offset + int(stop)) for start, stop in ranges]
    if isinstance(dataset, ConcatDataset):
        collected = []
        child_offset = offset
        for child in dataset.datasets:
            child_ranges = _collect_cache_shard_ranges(child, child_offset)
            if not child_ranges:
                return []
            collected.extend(child_ranges)
            child_offset += len(child)
        return collected
    return []


class CacheShardDistributedSampler(Sampler):
    """Assign cache shard ranges to ranks while keeping every rank balanced.

    Unlike a strided DistributedSampler, this sampler keeps rank reads local to
    a small set of mmap shards.  Shards are shuffled and samples are shuffled
    within each shard on every epoch.
    """

    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        ranges = _collect_cache_shard_ranges(dataset)
        if not ranges:
            raise ValueError("CacheShardDistributedSampler requires cache shard ranges")

        # If there are fewer files than ranks, split the largest ranges into
        # contiguous units.  This still limits each rank to a small file set.
        units = list(ranges)
        while len(units) < self.num_replicas:
            largest_index = max(range(len(units)), key=lambda i: units[i][1] - units[i][0])
            start, stop = units.pop(largest_index)
            if stop - start <= 1:
                units.insert(largest_index, (start, stop))
                break
            middle = start + (stop - start) // 2
            units.extend([(start, middle), (middle, stop)])
        self.units = units

        # Deterministic greedy bin packing gives all ranks the same assignment.
        assignments = [[] for _ in range(self.num_replicas)]
        loads = [0] * self.num_replicas
        for unit in sorted(units, key=lambda item: item[1] - item[0], reverse=True):
            target_rank = min(range(self.num_replicas), key=lambda rank: loads[rank])
            assignments[target_rank].append(unit)
            loads[target_rank] += unit[1] - unit[0]
        # A dataset can be smaller than the world size during smoke tests. In
        # that unavoidable case, duplicate one contiguous unit for empty ranks
        # instead of failing before the first step.
        non_empty_ranks = [rank for rank, assignment in enumerate(assignments) if assignment]
        for rank, assignment in enumerate(assignments):
            if assignment:
                continue
            source_rank = non_empty_ranks[rank % len(non_empty_ranks)]
            unit = assignments[source_rank][0]
            assignments[rank] = [unit]
            loads[rank] = unit[1] - unit[0]
        self.assignments = assignments
        self.num_samples = max(loads)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        units = list(self.assignments[self.rank])
        if self.shuffle and len(units) > 1:
            order = torch.randperm(len(units), generator=generator).tolist()
            units = [units[index] for index in order]

        indices = []
        for start, stop in units:
            if self.shuffle:
                local = torch.randperm(stop - start, generator=generator).tolist()
                indices.extend(start + index for index in local)
            else:
                indices.extend(range(start, stop))

        if not indices:
            raise RuntimeError(f"Cache shard sampler assigned no samples to rank {self.rank}")
        if len(indices) < self.num_samples:
            repeats = (self.num_samples + len(indices) - 1) // len(indices)
            indices = (indices * repeats)[:self.num_samples]
        else:
            indices = indices[:self.num_samples]
        return iter(indices)

PrefetchResult = Union[
    Tuple[Any, List[Any], int], # (iterator, prefetched_batches_list, epoch_fetched)
    Exception,
    type(StopIteration) # Use StopIteration class as signal
]

class BackgroundPrefetcher:
    """
    Aggressively prefetches the next epoch's iterator and *multiple* initial batches (k) in a background thread to completely hide cold 
    start latency and worker ramp-up time.
    
    Requires persistent_workers=True on the wrapped DataLoader.
    Requires the wrapped DataLoader to have a __len__ method.
    Does not handle skipping batches internally.
    """
    
    def __init__(self, loader: DataLoader, num_batches_to_prefetch: Optional[int] = None): 
        internal_len = -1
        try:
            internal_len = len(loader)
            if internal_len <= 0:
                raise ValueError("DataLoader length must be positive.")
        except TypeError:
            raise TypeError("BackgroundPrefetcher requires the wrapped DataLoader to have a __len__ method.")
             
        # logger.info(f"[BGPrefetcher] Initializing... Wrapped DataLoader len={internal_len}")
        
        if not hasattr(loader, 'persistent_workers') or not loader.persistent_workers:
            raise ValueError("BackgroundPrefetcher requires persistent_workers=True on the wrapped DataLoader.")
            
        self.loader = loader
        
        # Determine k: Number of batches to prefetch aggressively
        if num_batches_to_prefetch is None:
            # Default heuristic: prefetch factor + num workers might be enough
            self.num_batches_to_prefetch = getattr(loader, 'prefetch_factor', 2) + getattr(loader, 'num_workers', 1)
        else:
            self.num_batches_to_prefetch = max(1, num_batches_to_prefetch) # Must prefetch at least 1
        # logger.info(f"[BGPrefetcher] Will prefetch {self.num_batches_to_prefetch} batches per epoch.")

        self.epoch = 0 # Start at epoch 0
        self.prefetch_thread: Optional[threading.Thread] = None
        self.data_queue = queue.Queue(maxsize=1) 
        self._stop_event = threading.Event()

        # --- Initial blocking prefetch for Epoch 0 ---
        # logger.info(f"[BGPrefetcher] Performing initial (blocking) prefetch for starting Epoch {self.epoch} ({self.num_batches_to_prefetch} batches)... ---")
        start_time = time.time()
        self._prefetch_job(self.epoch) # Run directly, puts result in queue
        # logger.info(f"[BGPrefetcher] Initial prefetch done in {time.time() - start_time:.2f}s ---")
        # Result for Epoch 0 is now in the queue.

    def _prefetch_job(self, epoch_to_fetch: int):
        """
        Runs in background thread. Sets sampler, creates iterator, 
        fetches the first `k` batches, puts result in queue.
        """
        thread_id = threading.get_ident()
        if self._stop_event.is_set():
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Stop event set, skipping prefetch for Epoch {epoch_to_fetch}.")
            try: self.data_queue.put_nowait(StopIteration) 
            except queue.Full: pass 
            return
            
        prefetched_batches = []
        epoch_iter = None
        try:
            # 1. Set sampler epoch
            if isinstance(self.loader.sampler, Sampler) and hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(epoch_to_fetch)
                
            # 2. Create iterator and fetch first k batches (cold start bottleneck spread over k batches)
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Starting cold start for Epoch {epoch_to_fetch} (fetching {self.num_batches_to_prefetch} batches)... ---")
            start_cold_start = time.time()
            
            epoch_iter = iter(self.loader) # <-- Create iterator
            
            for i in range(self.num_batches_to_prefetch):
                batch = next(epoch_iter) # <-- Fetch batch (potentially blocking)
                prefetched_batches.append(batch)
                # Log timing for the very first batch fetch of the cold start
                if i == 0:
                    first_batch_time = time.time() - start_cold_start
                    # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Fetched *first* batch of Epoch {epoch_to_fetch} in {first_batch_time:.2f}s.")

            total_cold_start_time = time.time() - start_cold_start
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) *Finished* cold start for Epoch {epoch_to_fetch} (fetched {len(prefetched_batches)} batches) in {total_cold_start_time:.2f}s total. ---")
            
            # 3. Put successful result in queue (iterator is now advanced past the prefetched batches)
            self.data_queue.put((epoch_iter, prefetched_batches, epoch_to_fetch))
            
        except StopIteration:
            # This happens if the dataloader has less than k batches
            total_cold_start_time = time.time() - start_cold_start
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) DataLoader exhausted during prefetch for Epoch {epoch_to_fetch} after fetching {len(prefetched_batches)} batches (needed {self.num_batches_to_prefetch}). Total time: {total_cold_start_time:.2f}s.")
            # Put the partially fetched batches (if any) and signal exhaustion
            # We pass None for the iterator as it's exhausted
            self.data_queue.put((None, prefetched_batches, epoch_to_fetch)) 
             
        except Exception as e:
            import traceback
            # logger.info(f"[BGPrefetcher] (Thread {thread_id}) Prefetch failed for Epoch {epoch_to_fetch}: {e}\n{traceback.format_exc()} ---")
            self.data_queue.put(e)

    def _start_background_prefetch(self, epoch_to_fetch: int):
        """ Safely starts the background prefetch thread """
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            # logger.info(f"[BGPrefetcher] (Main) Joining previous prefetch thread (Epoch {epoch_to_fetch-1})...")
            self.prefetch_thread.join(timeout=60) # Increased timeout
            # if self.prefetch_thread.is_alive():
                # logger.info(f"[BGPrefetcher] (Main) Previous prefetch thread did not join cleanly!")
            # else:
            #      # logger.info(f"[BGPrefetcher] (Main) Previous prefetch thread joined.")
            
        if self._stop_event.is_set():
            # logger.info(f"[BGPrefetcher] (Main) Stop event set, not starting prefetch for Epoch {epoch_to_fetch}.")
            return

        # logger.info(f"[BGPrefetcher] (Main) Starting background prefetch thread for Epoch {epoch_to_fetch}... ---")
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_job, 
            args=(epoch_to_fetch,),
            daemon=True
        )
        self.prefetch_thread.start()

    def __iter__(self):
        """ Trainer calls this at the start of each epoch (e.g., Epoch N) """
        
        current_epoch_requested = self.epoch 
        # logger.info(f"[BGPrefetcher] (Main) __iter__ called for Epoch {current_epoch_requested}. Waiting for prefetched data... ---")
        
        # 1. Block and get the result for the CURRENT epoch (Epoch N)
        try:
            data: PrefetchResult = self.data_queue.get(timeout=1800) # 30 minutes timeout
        except queue.Empty:
            # logger.info(f"Timeout waiting for prefetched data for Epoch {current_epoch_requested}.")
            raise TimeoutError(f"Timeout waiting for prefetched data for Epoch {current_epoch_requested}.")

        # Check for error signals
        if isinstance(data, Exception): 
            # logger.info(f"[BGPrefetcher] (Main) Received exception from prefetch job for Epoch {current_epoch_requested}.")
            raise data
        # StopIteration signal is handled within the generator now
            
        # Unpack successful prefetch result (iterator might be None if exhausted during prefetch)
        current_iter, prefetched_batches, fetched_epoch = data
        # logger.info(f"[BGPrefetcher] (Main) Successfully received {len(prefetched_batches)} prefetched batches for Epoch {fetched_epoch}. ---")

        if fetched_epoch != current_epoch_requested:
            # logger.info(f"[BGPrefetcher] Epoch mismatch! Expected {current_epoch_requested}, got {fetched_epoch}. Adjusting internal epoch counter.")
            current_epoch_requested = fetched_epoch # Trust the fetched data

        # 2. Start prefetching the *NEXT* epoch (N+1) in the background *NOW*
        next_epoch_to_prefetch = current_epoch_requested + 1
        self._start_background_prefetch(next_epoch_to_prefetch)

        # 3. Define and return the generator for the CURRENT epoch (Epoch N)
        # Pass the prefetched batches and the (potentially None) iterator
        return self._generator(prefetched_batches, current_iter, current_epoch_requested)


    def _generator(self, prefetched_list: List[Any], iterator: Optional[Any], epoch_num: int):
        """ Generator yields prefetched batches, then remaining batches for one epoch. """
        total_yielded_this_epoch = 0
        try:
            # logger.info(f"[BGPrefetcher] (Generator) Starting Epoch {epoch_num} with {len(prefetched_list)} prefetched batches.")
            
            # 1. Yield the prefetched batches first
            for i, batch in enumerate(prefetched_list):
                # # logger.info(f"[BGPrefetcher] (Generator) Yielding prefetched batch {i} for Epoch {epoch_num}...")
                yield batch
                total_yielded_this_epoch += 1

            # 2. Yield remaining batches from the iterator (if it exists and wasn't exhausted)
            if iterator is not None:
                # # logger.info(f"[BGPrefetcher] (Generator) Yielding remaining batches for Epoch {epoch_num}...")
                for batch in iterator:
                    yield batch
                    total_yielded_this_epoch += 1
            # else:
                # logger.info(f"[BGPrefetcher] (Generator) Iterator was already exhausted during prefetch for Epoch {epoch_num}.")


            # logger.info(f"\n[BGPrefetcher] (Generator) Epoch {epoch_num} exhausted after {total_yielded_this_epoch} batches.")

        except Exception as e:
             # logger.info(f"[BGPrefetcher] (Generator) Error during iteration for Epoch {epoch_num}: {e}", exc_info=True)
             raise e
        finally:
            # Update epoch counter AFTER the generator is exhausted
            # logger.info(f"[BGPrefetcher] (Generator) Incrementing epoch counter after Epoch {epoch_num} finished.")
            self.epoch = epoch_num + 1

    def __len__(self):
        # Must return the correct length for Trainer's calculations
        return len(self.loader) 
    
    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def batch_sampler(self):
        return getattr(self.loader, 'batch_sampler', None)

    def set_epoch(self, epoch: int):
        # logger.info(f"[BGPrefetcher] (Main) Trainer called set_epoch({epoch}). Internal epoch is {self.epoch}. (Call ignored by prefetcher internal logic)")
        pass

    def shutdown(self):
        # logger.info("[BGPrefetcher] Shutting down...")
        self._stop_event.set()
        # Try to unblock queue if thread is waiting to put
        try: self.data_queue.put(StopIteration, block=False) 
        except queue.Full: # logger.info("[BGPrefetcher] Queue full during shutdown signal.")
            pass
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            # logger.info("[BGPrefetcher] Joining background thread...")
            self.prefetch_thread.join(timeout=10) 
            # if self.prefetch_thread.is_alive():
                # logger.info("[BGPrefetcher] Background thread did not join cleanly.")
        # Drain queue
        while not self.data_queue.empty():
            try: self.data_queue.get_nowait()
            except queue.Empty: break
        # logger.info("[BGPrefetcher] Shutdown complete.")

class MultiplexedDataLoader:
    """
    A DataLoader multiplexer that iterates over multiple DataLoaders.
    
    Strategies:
    - Training: Weighted Random Selection using Numpy. Loaders are chosen based on their cumulative weights.
                Loaders are restarted infinitely upon exhaustion.
    - Evaluation: Deterministic Round-Robin. Iterates until ALL loaders are exhausted.
    """
    def __init__(self, loaders, loader_weights=None, is_training=True):
        self.loaders = loaders
        self.is_training = is_training
        self.num_loaders = len(loaders)
        
        # Handle weights for loader selection
        if loader_weights is None or not is_training:
            # Use uniform weights for evaluation or if not provided
            self.loader_weights = [1.0] * self.num_loaders
        else:
            self.loader_weights = loader_weights
            
        assert len(self.loader_weights) == self.num_loaders, "Length of weights must match number of loaders"

        # Calculate approximate length for progress bars
        self._len = 0
        if not is_training:
            # For eval, the length is the sum of all sub-loaders
            for l in loaders:
                if hasattr(l, '__len__'): self._len += len(l)
        else:
            # For training, return the max length of sub-loaders as a proxy
            lens = [len(l) for l in loaders if hasattr(l, '__len__')]
            self._len = max(lens) if lens else 0

    def __len__(self):
        return self._len

    def __iter__(self):
        # Create iterators for all loaders
        iterators = [iter(l) for l in self.loaders]
        
        # Track active indices and their weights (dynamic for eval mode)
        active_indices = list(range(self.num_loaders))
        active_weights = list(self.loader_weights)
        
        # Cycle pointer for Round-Robin (Eval only)
        cycle_ptr = 0
        
        while len(active_indices) > 0:
            # === Selection Strategy ===
            if self.is_training:
                # Weighted Random Selection using Numpy
                # 1. Convert to numpy array
                w_arr = np.array(active_weights, dtype=np.float64)
                # 2. Normalize to probabilities (sum must be 1.0 for np.random.choice)
                if w_arr.sum() == 0:
                    probs = None # Let numpy handle uniform if sum is 0 (edge case)
                else:
                    probs = w_arr / w_arr.sum()
                
                # 3. Select an INDEX from the active_indices list
                # np.random.choice returns a scalar here
                chosen_list_idx = np.random.choice(len(active_indices), p=probs)
                
            else:
                # Deterministic Round-Robin (No random needed)
                chosen_list_idx = cycle_ptr % len(active_indices)
            
            # Map back to the original loader index
            loader_idx = active_indices[chosen_list_idx]
            iterator = iterators[loader_idx]
            
            try:
                batch = next(iterator)
                yield batch
                
                # Advance cycle pointer (only affects Eval)
                cycle_ptr += 1
                
            except StopIteration:
                if self.is_training:
                    # === Training: Infinite Restart ===
                    # Re-initialize the exhausted loader
                    iterators[loader_idx] = iter(self.loaders[loader_idx])
                    
                    # Immediately retry fetching from this re-initialized loader
                    continue 
                else:
                    # === Evaluation: Drain and Remove ===
                    # Remove this loader from active pool
                    active_indices.pop(chosen_list_idx)
                    active_weights.pop(chosen_list_idx)

def get_dataloader(train_dataset, val_dataset=None, processor=None, collator=None, args=None):
    """
    Create DataLoader from single dataset or multiple datasets.
    
    Args:
        train_dataset: Single dataset or list of datasets
        val_dataset: Optional validation dataset
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
    
    Returns:
        (train_loader, eval_loader)
    """
    if isinstance(train_dataset, list):
        # Multiple datasets - use torchdata for mixing
        
        train_loader = _create_mixed_dataloader(train_dataset, processor, collator, args)
        
        # Handle validation dataset
        eval_loader = None
        if val_dataset is not None:
            if isinstance(val_dataset, list):
                eval_loader = _create_mixed_dataloader(val_dataset, processor, collator, args, is_training=False)
            else:
                eval_loader = _create_single_dataloader(val_dataset, processor, collator, args, is_training=False)
        
        return train_loader, eval_loader
    
    else:
        # Single dataset - use existing logic
        train_loader = _create_single_dataloader(train_dataset, processor, collator, args, is_training=True)
        
        # Handle validation dataset
        eval_loader = None
        if val_dataset is not None:
            eval_loader = _create_single_dataloader(val_dataset, processor, collator, args, is_training=False)
        
        return train_loader, eval_loader

def is_rlds_data(ds):
    return isinstance(ds, dl.DLataset)

def is_map_data(dataset):
    return hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')

def is_iter_data(dataset):
    return hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))

def _create_single_dataloader(dataset, processor, collator, args, is_training=True, sample_weights=None):
    """Create DataLoader for a single dataset (map-style or iterable)"""
    # Identify the type of the dataset: iter or map
    if is_map_data(dataset):
        wrapped_data = WrappedDataset(dataset, processor)

        # 1. Sampler Logic
        sampler = None
        if sample_weights is None:
            if is_distributed():
                cache_ranges = _collect_cache_shard_ranges(wrapped_data.dataset)
                if cache_ranges:
                    sampler = CacheShardDistributedSampler(
                        wrapped_data.dataset,
                        shuffle=is_training,
                        seed=getattr(args, 'seed', 0),
                    )
                else:
                    sampler = DistributedSampler(wrapped_data, shuffle=is_training)
            else:
                sampler = None
        else:
            if is_training:
                # Case A: Training with Weights
                # Ensure weights are a Double Tensor for precision
                weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
                if is_distributed():
                    # 
                    # Simulate DDP behavior with WeightedRandomSampler.
                    # We restrict num_samples so each GPU processes a fraction of the total.
                    world_size = dist.get_world_size() 
                    rank = dist.get_rank()
                    num_samples = int(np.ceil(len(wrapped_data) / world_size))
                    
                    # CRITICAL: Use a distinct seed per rank. 
                    # Otherwise, all GPUs will sample the exact same indices.
                    sampler_generator = torch.Generator()
                    sampler_generator.manual_seed(getattr(args, 'seed', 0) + rank)
                    
                    sampler = WeightedRandomSampler(
                        weights=weights_tensor,
                        num_samples=num_samples,
                        replacement=True,
                        generator=sampler_generator
                    )
                else:
                    # Single GPU Standard Weighted Sampling
                    sampler = WeightedRandomSampler(
                        weights=weights_tensor,
                        num_samples=len(wrapped_data),
                        replacement=True
                    )
            else:
                sampler = DistributedSampler(wrapped_data, shuffle=is_training) if is_distributed() else None
        
        generator = None
        if sampler is None:  
            seed = getattr(args, 'seed', 0)
            generator = torch.Generator()
            generator.manual_seed(seed)
        
        # Worker init func
        def worker_init_fn(worker_id):
            import random
            seed = getattr(args, 'seed', 0)
            rank = dist.get_rank() if dist.is_initialized() else 0
            worker_seed = seed + worker_id + (rank * 1000) # ensure different GPU seeds
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        persistent_workers = getattr(args, 'dataloader_persistent_workers', False) if args.dataloader_num_workers>0 else False
        prefetch_factor = getattr(args, 'dataloader_prefetch_factor', 2) if args.dataloader_num_workers>0 else None
        
        # Build DataLoader kwargs
        dataloader_kwargs = {
            'dataset': wrapped_data,
            'batch_size': args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size,
            'shuffle': (is_training and sampler is None),
            'sampler': sampler,
            'num_workers': args.dataloader_num_workers,
            'collate_fn': collator,
            'drop_last': is_training,
            'pin_memory': args.dataloader_pin_memory,
            'generator': generator,
        }
        
        # Only add multiprocessing-specific parameters when num_workers > 0
        if args.dataloader_num_workers > 0:
            dataloader_kwargs['persistent_workers'] = persistent_workers
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            if is_training:
                dataloader_kwargs['worker_init_fn'] = worker_init_fn
        
        loader = DataLoader(**dataloader_kwargs)
        if getattr(args, 'background_prefetch', False):
            loader = BackgroundPrefetcher(loader, getattr(args, 'dataloader_prefetch_factor', 2))
    elif is_iter_data(dataset):
        if hasattr(dataset, 'dataset') and is_rlds_data(dataset.dataset):
            # RLDS dataset
            # set tf data options here
            import tensorflow as tf
            dataset.dataset = dataset.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            dataset.dataset = dataset.dataset.with_ram_budget(1)
            wrapped_data = WrappedIterableDataset(dataset, processor)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            loader = DataLoader(
                wrapped_data,  
                batch_size=batch_size,
                num_workers=args.dataloader_num_workers,
                collate_fn=collator,
                persistent_workers=True,
            )
        else:
            # Pytorch Iterable dataset
            # Wrap iterable dataset with processor
            try:
                from torchdata.datapipes.iter import IterableWrapper
            except ImportError:
                raise ImportError(
                    "torchdata is required for iterable dataset support. "
                    "Install it with: pip install torchdata"
                )
            
            wrapped_data = WrappedIterableDataset(dataset, processor)
            pipe = IterableWrapper(wrapped_data, deepcopy=False)
            # For iterable datasets, we cannot use DistributedSampler
            pipe = pipe.sharding_filter()
            pipe = pipe.cycle()
            buffer_size = getattr(args, 'shuffle_buffer_size', 1000)
            pipe = pipe.shuffle(buffer_size=buffer_size)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            #### Prefetch is handled by DataLoader
            loader = DataLoader(
                pipe,
                batch_size=batch_size,  
                # num_workers=args.dataloader_num_workers,
                collate_fn=collator, 
                pin_memory=args.dataloader_pin_memory,
                # persistent_workers=args.dataloader_num_workers>0,
                drop_last=True,
            )
    else:
        raise ValueError("Dataset must be either map-style or iterable.")
    return loader

def _create_mixed_dataloader(datasets, processor, collator, args, is_training=True):
    """
    Create DataLoader for multiple datasets using torchdata pipeline.
    
    Args:
        datasets: List of datasets (can be map-style or iterable)
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
        is_training: Whether this is for training (affects shuffle, drop_last)
    
    Returns:
        DataLoader with mixed pipeline
    """
    # Create DataLoader for MapStyleDataset, IterableDataset, and RLDSDataset, respectively
    all_map_datasets, all_iter_datasets, all_rlds_datasets = [], [], []
    for data in datasets:
        if is_map_data(data): all_map_datasets.append(data)
        elif is_iter_data(data):
            if is_rlds_data(data):
                all_rlds_datasets.append(data)
            else:
                all_iter_datasets.append(data)
        else:
            raise TypeError("Dataset must be either map-style or iterable.")
    # Mix map-style datasets using ConcatDataset
    map_loader = None
    if len(all_map_datasets)>0:
        if hasattr(all_map_datasets[0], '__weight__'):
            sample_weights = []
            for ds in all_map_datasets:
                sample_weights.extend([ds.__weight__] * len(ds))
            map_loader_weight = sum([ds.__weight__ for ds in all_map_datasets])
        else:
            sample_weights = None
            map_loader_weight = 1.0
        mixed_map_data = torch.utils.data.ConcatDataset(all_map_datasets)
        map_loader = _create_single_dataloader(mixed_map_data, processor, collator, args, is_training=is_training, sample_weights=sample_weights)
    
    # mix iterable datasets using huggingface's datasets
    iter_loader = None
    if len(all_iter_datasets)>0:
        from datasets import interleave_datasets
        if hasattr(all_iter_datasets[0], '__weight__'):
            sample_weights = [ds.__weight__ for ds in all_iter_datasets]
            sample_weights_sum = sum(sample_weights)
            sample_weights = [w / sample_weights_sum for w in sample_weights]
            iter_loader_weight = sample_weights_sum
        else:
            sample_weights = None
            iter_loader_weight = 1.0
        mixed_iter_data = interleave_datasets(all_iter_datasets, probabilities=sample_weights)
        iter_loader = _create_single_dataloader(mixed_iter_data, processor, collator, args, is_training=is_training)
    
    # mix rlds datasets using tf.data
    rlds_loader = None
    if len(all_rlds_datasets)>0:
        import dlimp as dl
        if hasattr(all_rlds_datasets[0], '__weight__'):
            sample_weights = [ds.__weight__ for ds in all_rlds_datasets]
            rlds_loader_weight = sum([ds.__weight__ for ds in all_rlds_datasets])
        else:
            sample_weights = None
            rlds_loader_weight = 1.0
        mixed_rlds_data_raw = dl.DLataset.sample_from_datasets([ds.dataset if hasattr(ds, 'dataset') else ds for ds in all_rlds_datasets], weights=sample_weights)
        # _create_single_dataloader has specific logic for rlds_data
        rlds_loader = _create_single_dataloader(mixed_rlds_data_raw, processor, collator, args, is_training=is_training)
    
    # Combine all loaders
    loaders_available = []
    loaders_weights = []
    if map_loader is not None:
        loaders_available.append(map_loader)
        loaders_weights.append(map_loader_weight)
    if iter_loader is not None:
        loaders_available.append(iter_loader)
        loaders_weights.append(iter_loader_weight)
    if rlds_loader is not None:
        loaders_available.append(rlds_loader)
        loaders_weights.append(rlds_loader_weight)
    
    if len(loaders_available) == 1:
        return loaders_available[0]
    elif len(loaders_available) > 1:
        # If multiple types of loaders are present, we need a multiplexer.
        return MultiplexedDataLoader(loaders_available, loaders_weights, is_training=is_training)
    else:
        raise ValueError("No loaders available.")
    
