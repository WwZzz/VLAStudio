"""Public API remains lightweight and legacy blocking inference still works."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class ExecutionAPITests(unittest.TestCase):
    def test_public_api_lazy_loads_execution(self):
        code = "import sys, vlastudio; assert callable(vlastudio.open_execution); assert callable(vlastudio.connect_target); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_public_yaml_entrypoint_loads_config_without_mutating_it(self):
        import vlastudio
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'execution.yaml'
            path.write_text('control_hz: 20\n')
            with patch('vlastudio.deploy.execution_session.ExecutionSession') as session:
                vlastudio.open_execution(path, output_dir=directory)
            self.assertEqual(session.call_args.args, ({'control_hz': 20},))

    def test_legacy_worker_without_context_and_blocking_action_manager(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '')
        from vlastudio.deploy.inference import start_inference_process, stop_inference_process
        from vlastudio.deploy.action_manager import BasicActionManager
        from execution_session_fixtures import make_observation
        with tempfile.TemporaryDirectory() as directory:
            context = start_inference_process(directory, device='cpu',
                policy_loader='execution_session_fixtures:load_policy',
                startup_timeout_s=30, shm_prefix='legacy_acceptance')
            try:
                self.assertIsNone(context.context_writer)
                context.update_obs(make_observation(7), 1)
                manager = BasicActionManager()
                manager.set_inference_context(context)
                action = manager.select_action()
                self.assertIsNotNone(action)
                self.assertEqual(manager.t, 1)
            finally:
                stop_inference_process(context)


if __name__ == '__main__':
    unittest.main()
