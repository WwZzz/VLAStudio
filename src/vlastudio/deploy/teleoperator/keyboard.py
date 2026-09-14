"""
Keyboard Teleoperator - Generic configurable keyboard control
Uses GUI window to capture keyboard events (works in subprocess)
"""

import tkinter as tk
import numpy as np
import multiprocessing as mp
from typing import Optional, List, Dict

from vlastudio.deploy.teleoperator.base import BaseTeleopDevice


class Keyboard(BaseTeleopDevice):
    """
    Generic keyboard teleoperation with configurable key mappings.
    
    Uses a Tkinter window to capture keyboard events, which works in subprocesses
    (unlike pynput which requires the main thread).
    
    Key mappings format:
        key_mappings = [
            {
                "name": "X (forward/back)",      # Display name
                "key_positive": "w",             # Key for positive delta
                "key_negative": "s",             # Key for negative delta
                "scale": 1.0,                    # Scale factor
            },
            ...
        ]
    
    Each mapping defines one action dimension.
    """

    def __init__(
        self,
        name: str = "keyboard",
        max_size_mb: int = 1,
        fps: float = 50.0,
        title: str = "Keyboard Teleop",
        key_mappings: Optional[List[Dict]] = None,
        reset_key: str = "0",
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize keyboard teleoperator
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency
            title: Window title
            key_mappings: List of key mapping dicts, each with:
                - name: Display name for this action
                - key_positive: Key for positive delta
                - key_negative: Key for negative delta  
                - scale: Scale factor (default 1.0)
            reset_key: Key to reset all actions to zero
            debug: Enable debug output
        """
        super().__init__(name, max_size_mb=max_size_mb, fps=fps)
        
        self.title = title
        self.reset_key = reset_key.lower()
        self.debug = debug
        
        # Default key mappings if not provided
        if key_mappings is None:
            key_mappings = [
                {"name": "Action 0", "key_positive": "w", "key_negative": "s", "scale": 1.0},
            ]
        
        self.key_mappings = key_mappings
        self.action_dim = len(key_mappings)
        
        # Will be initialized in start()
        self._action_values = None
        self._gui_proc = None

    def _build_instructions(self) -> str:
        """Build instruction text from key mappings"""
        lines = []
        for m in self.key_mappings:
            name = m.get("name", "Action")
            key_pos = str(m.get("key_positive", "?")).upper()
            key_neg = str(m.get("key_negative", "?")).upper()
            lines.append(f"  {name}: {key_pos}/{key_neg}")
        lines.append(f"\n  Reset: {str(self.reset_key).upper()}")
        return "\n".join(lines)

    def _keyboard_gui_process(self, shared_array, key_mappings, reset_key, title, action_dim, debug=False):
        """GUI process that captures keyboard events"""
        print(f"[Keyboard GUI] Process started, debug={debug}", flush=True)
        
        # Set of currently pressed keys
        pressed_keys = set()
        
        # Create main window
        root = tk.Tk()
        root.title(title)
        root.geometry("500x600")
        root.configure(bg='#2b2b2b')
        
        # Make window focusable
        root.focus_force()
        
        # --- Header ---
        header = tk.Label(
            root, 
            text=title,
            font=('Consolas', 14, 'bold'),
            fg='white',
            bg='#2b2b2b'
        )
        header.pack(pady=10)
        
        # --- Instructions Frame ---
        instr_frame = tk.Frame(root, bg='#3c3c3c', padx=10, pady=10)
        instr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Build instructions from key mappings
        instr_lines = ["Key Mappings:"]
        for m in key_mappings:
            name = m.get("name", "Action")
            key_pos = str(m.get("key_positive", "?")).upper()
            key_neg = str(m.get("key_negative", "?")).upper()
            instr_lines.append(f"  {name}: {key_pos} (+) / {key_neg} (-)")
        instr_lines.append(f"\n  Reset all: {str(reset_key).upper()}")
        instructions = "\n".join(instr_lines)
        
        instr_label = tk.Label(
            instr_frame,
            text=instructions,
            font=('Consolas', 11),
            fg='#00ff00',
            bg='#3c3c3c',
            justify=tk.LEFT
        )
        instr_label.pack()
        
        # --- Status Label (shows currently pressed keys) ---
        status_label = tk.Label(
            root,
            text="Keys: (none)",
            font=('Consolas', 12),
            fg='yellow',
            bg='#2b2b2b'
        )
        status_label.pack(pady=10)
        
        # --- Action Display Frame ---
        action_frame = tk.Frame(root, bg='#3c3c3c', padx=10, pady=10)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        action_title = tk.Label(
            action_frame,
            text="Current Action:",
            font=('Consolas', 12, 'bold'),
            fg='white',
            bg='#3c3c3c'
        )
        action_title.pack(anchor=tk.W)
        
        # Create labels for each action component
        action_labels = []
        for m in key_mappings:
            name = m.get("name", "Action")
            frame = tk.Frame(action_frame, bg='#3c3c3c')
            frame.pack(fill=tk.X, pady=2)
            
            name_label = tk.Label(
                frame,
                text=f"  {name}:",
                font=('Consolas', 10),
                fg='#aaaaaa',
                bg='#3c3c3c',
                width=18,
                anchor=tk.W
            )
            name_label.pack(side=tk.LEFT)
            
            value_label = tk.Label(
                frame,
                text="+0.00",
                font=('Consolas', 12, 'bold'),
                fg='#00ff00',
                bg='#3c3c3c',
                width=10
            )
            value_label.pack(side=tk.LEFT)
            action_labels.append(value_label)
        
        # --- Log Text Box ---
        log_frame = tk.Frame(root, bg='#2b2b2b')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_title = tk.Label(
            log_frame,
            text="Action Log:",
            font=('Consolas', 10),
            fg='white',
            bg='#2b2b2b',
            anchor=tk.W
        )
        log_title.pack(anchor=tk.W)
        
        log_text = tk.Text(
            log_frame,
            height=8,
            font=('Consolas', 9),
            fg='#00ff00',
            bg='#1a1a1a',
            state=tk.DISABLED
        )
        log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log counter for limiting log entries
        log_counter = [0]
        max_log_lines = 100
        
        def log_action(action):
            """Add action to log"""
            log_counter[0] += 1
            if log_counter[0] % 10 != 0:  # Only log every 10th action
                return
            
            log_text.config(state=tk.NORMAL)
            
            # Format action string
            action_strs = [f"{a:+.2f}" for a in action]
            action_str = f"[{log_counter[0]:5d}] [{', '.join(action_strs)}]\n"
            
            log_text.insert(tk.END, action_str)
            
            # Keep only last N lines
            lines = int(log_text.index('end-1c').split('.')[0])
            if lines > max_log_lines:
                log_text.delete('1.0', f'{lines - max_log_lines}.0')
            
            log_text.see(tk.END)
            log_text.config(state=tk.DISABLED)
        
        def on_key_press(event):
            key = event.keysym.lower()
            if debug:
                print(f"[Keyboard] Key press: keysym='{event.keysym}' -> '{key}'", flush=True)
            pressed_keys.add(key)
        
        def on_key_release(event):
            key = event.keysym.lower()
            pressed_keys.discard(key)
        
        def update_action():
            """Update action based on pressed keys"""
            action = np.zeros(action_dim, dtype=np.float64)
            
            # Debug: print pressed keys periodically
            if debug and pressed_keys:
                print(f"[Keyboard] Pressed keys: {pressed_keys}", flush=True)
            
            # Process each key mapping
            for i, m in enumerate(key_mappings):
                key_pos = str(m.get("key_positive", "")).lower()
                key_neg = str(m.get("key_negative", "")).lower()
                scale = m.get("scale", 1.0)
                
                # Check if key matches (handle both single char and keysym names)
                pos_match = key_pos in pressed_keys
                neg_match = key_neg in pressed_keys
                
                if pos_match:
                    action[i] += scale
                if neg_match:
                    action[i] -= scale
            
            # Reset
            if reset_key in pressed_keys:
                action = np.zeros(action_dim, dtype=np.float64)
            
            # Write to shared array
            for i in range(action_dim):
                shared_array[i] = action[i]
            
            # Update status label
            if pressed_keys:
                keys_str = ', '.join(sorted(pressed_keys))
                status_label.config(text=f"Keys: {keys_str}")
            else:
                status_label.config(text="Keys: (none)")
            
            # Update action display
            for i, label in enumerate(action_labels):
                val = action[i]
                label.config(text=f"{val:+.2f}")
                if val > 0:
                    label.config(fg='#00ff00')  # Green for positive
                elif val < 0:
                    label.config(fg='#ff6666')  # Red for negative
                else:
                    label.config(fg='#888888')  # Gray for zero
            
            # Log action if non-zero
            if np.any(action != 0):
                log_action(action)
            
            # Schedule next update
            root.after(20, update_action)  # 50 Hz
        
        # Bind keyboard events
        root.bind('<KeyPress>', on_key_press)
        root.bind('<KeyRelease>', on_key_release)
        
        # Start update loop
        root.after(20, update_action)
        
        # Focus instruction
        focus_label = tk.Label(
            root,
            text="(Click here or press any key to focus)",
            font=('Consolas', 9),
            fg='#888888',
            bg='#2b2b2b'
        )
        focus_label.pack(pady=5)
        
        root.mainloop()

    def start(self):
        """Start the keyboard teleoperator with GUI in a subprocess"""
        import signal
        from vlastudio.deploy.utils import RateLimiter
        
        if self.debug:
            print(f"[Keyboard] Starting, name={self.name}")
        
        # Create shared memory for output
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Create shared array for inter-process communication with GUI
        self._action_values = mp.Array('d', self.action_dim)
        
        # Start the GUI process
        self._gui_proc = mp.Process(
            target=self._keyboard_gui_process,
            args=(self._action_values, self.key_mappings, self.reset_key, self.title, self.action_dim, self.debug)
        )
        self._gui_proc.daemon = True
        self._gui_proc.start()
        
        if self.debug:
            print(f"[Keyboard] GUI process started, pid={self._gui_proc.pid}")
        
        # Setup signal handler to clean up GUI process on termination
        def cleanup_handler(signum, frame):
            self.close()
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        
        # Main loop: read from mp.Array and write to shared memory
        self.is_running = True
        rate_limiter = RateLimiter()
        loop_count = 0
        try:
            while self.is_running:
                data = self.get_data()
                if data is not None:
                    action = self.convert_data_to_action(data)
                    self.write_data_to_shm({"action": action})
                    loop_count += 1
                    if self.debug and loop_count % 50 == 1:
                        print(f"[Keyboard] Writing action: {action}")
                rate_limiter.sleep(self.fps)
        finally:
            self.close()

    def get_data(self) -> Optional[dict]:
        """Read the current action values from the shared array"""
        if self._action_values is None:
            return None
        arr = np.frombuffer(self._action_values.get_obj()).copy()
        return {"action": arr}

    def convert_data_to_action(self, data: dict) -> np.ndarray:
        """Convert data to action (already in action format)"""
        return data.get("action", np.zeros(self.action_dim))

    def close(self):
        """Close the teleoperation device"""
        self.is_running = False
        if self._gui_proc is not None and self._gui_proc.is_alive():
            self._gui_proc.terminate()
            self._gui_proc.join(timeout=1.0)
            # Force kill if still alive
            if self._gui_proc.is_alive():
                self._gui_proc.kill()
                self._gui_proc.join(timeout=0.5)
        super().close()


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import time
    from vlastudio.deploy.base import start_device
    from vlastudio.deploy.shm_utils import SharedMemoryChannel

    # Example: 3D position control
    device_config = {
        "type": "deploy.teleoperator.keyboard.Keyboard",
        "name": "keyboard",
        "args": {
            "name": "keyboard",
            "title": "3D Position Control",
            "fps": 50.0,
            "key_mappings": [
                {"name": "X (fwd/back)", "key_positive": "w", "key_negative": "s", "scale": 1.0},
                {"name": "Y (left/right)", "key_positive": "a", "key_negative": "d", "scale": 1.0},
                {"name": "Z (up/down)", "key_positive": "r", "key_negative": "f", "scale": 1.0},
            ],
            "reset_key": "0",
            "debug": True,
        }
    }

    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()

    time.sleep(1.0)
    print("Reading from SHM (Ctrl+C to stop)...")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=10.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                print(f"  action: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        print("Done.")
