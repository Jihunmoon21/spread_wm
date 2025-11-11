import numpy as np
import pyflex

from .scenes import *


class FlexScene:
    def __init__(self):
        self.obj = None
        self.env_idx = None
        self.scene_params = None

        self.property_params = None
        self.clusters = None

    def set_scene(self, obj, obj_params):
        self.obj = obj

        if self.obj == "rope":
            self.env_idx = 26
            self.scene_params, self.property_params = rope_scene(obj_params)
        elif self.obj == "granular":
            self.env_idx = 7  # GranularPile is at index 7 (after Softgym scenes)
            self.scene_params, self.property_params = granular_scene(obj_params)
        elif self.obj == "cloth":
            self.env_idx = 29
            self.scene_params, self.property_params = cloth_scene(obj_params)
        else:
            raise ValueError("Unknown Scene.")

        assert self.env_idx is not None
        assert self.scene_params is not None
        
        # Ensure scene_params is properly formatted
        scene_params_f32 = self.scene_params.astype(np.float32)
        
        # Use 3-arg binding: (env_idx: int, scene_params: float32[n], flags: int)
        # For granular scenes, ensure parameters are valid
        if self.obj == "granular":
            # Validate granular scene parameters
            assert len(scene_params_f32) > 0, "granular scene_params is empty"
            # Note: env_idx is now 9 (GranularPile moved to index 9 in PyFlex)
            
            # Debug: Print scene parameters before setting
            print(f"Setting granular scene with {len(scene_params_f32)} parameters")
            print(f"First few params: {scene_params_f32[:10]}")
            try:
                cx, cy, cz = float(scene_params_f32[5]), float(scene_params_f32[6]), float(scene_params_f32[7])
                print(f"[SCENE DEBUG] pos_granular(center) to C++: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
            except Exception as _:
                pass
            
            # Verify no NaN or Inf values
            if np.any(np.isnan(scene_params_f32)) or np.any(np.isinf(scene_params_f32)):
                raise ValueError(f"Invalid scene_params (NaN/Inf detected): {scene_params_f32}")
        
        # Flush stdout to ensure debug messages are printed before potential crash
        import sys
        sys.stdout.flush()
        
        # For granular scenes, ensure scene_params array is contiguous and properly aligned
        if self.obj == "granular":
            scene_params_f32 = np.ascontiguousarray(scene_params_f32, dtype=np.float32)
            # Ensure array is not empty and has valid memory layout
            assert scene_params_f32.flags['C_CONTIGUOUS'], "scene_params must be C-contiguous"
            assert scene_params_f32.flags['WRITEABLE'], "scene_params must be writeable"
        
        # For granular scenes on new GPUs, there may be a compatibility issue
        # Try to work around by using a signal handler to catch potential crashes
        if self.obj == "granular":
            import signal
            
            crash_occurred = [False]
            
            def crash_handler(signum, frame):
                crash_occurred[0] = True
                print(f"\nCRITICAL: Segmentation fault detected during pyflex.set_scene(35, ...)")
                print("This indicates a PyFlex binary compatibility issue with your GPU.")
                print("The PyFlex C++ code for env_idx 35 may not support RTX 5060 Ti architecture.")
                raise RuntimeError("PyFlex set_scene crashed - likely GPU architecture incompatibility")
            
            # Set up signal handler for SIGSEGV (segmentation fault)
            old_handler = signal.signal(signal.SIGSEGV, crash_handler)
            timeout_triggered = [False]

            def timeout_handler(signum, frame):
                timeout_triggered[0] = True
                raise TimeoutError("pyflex.set_scene timed out")

            old_alarm_handler = signal.signal(signal.SIGALRM, timeout_handler)
        else:
            old_alarm_handler = None
        
        try:
            # Critical: Call set_scene with explicit type casting
            env_idx_int = int(self.env_idx)
            flags_int = int(0)
            
            # Double-check parameters before call
            if self.obj == "granular":
                print("[DEBUG] Preparing to call pyflex.set_scene (granular)")
                print(f"Calling pyflex.set_scene({env_idx_int}, array[{len(scene_params_f32)}], {flags_int})")
                sys.stdout.flush()
            
            # Try to call with a simple attempt first
            try:
                if self.obj == "granular":
                    print("[DEBUG] About to execute pyflex.set_scene(...)")
                    sys.stdout.flush()
                    signal.setitimer(signal.ITIMER_REAL, 5.0)
                # This call may hang indefinitely - no Python-level solution exists
                pyflex.set_scene(env_idx_int, scene_params_f32, flags_int)
                if self.obj == "granular":
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    print("[DEBUG] Returned from pyflex.set_scene(...) without exception")
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("\n⚠️  pyflex.set_scene() was interrupted - likely hung")
                print("   This confirms a blocking issue in PyFlex C++ code")
                print("   Solution: PyFlex needs to be fixed or rebuilt for your GPU")
                raise
            except TimeoutError:
                if self.obj == "granular":
                    print("\n⚠️  pyflex.set_scene() timed out after 60 seconds - aborting scene setup")
                raise
            
            # If we get here, set_scene succeeded
            if self.obj == "granular":
                print("✓ pyflex.set_scene() completed successfully")
                sys.stdout.flush()
                
        except (RuntimeError, Exception) as e:
            if self.obj == "granular":
                print(f"\nFailed to set granular scene: {e}")
                print("\nPossible solutions:")
                print("1. Rebuild PyFlex bindings for your GPU architecture")
                print("2. Use a different GPU that supports this PyFlex version")
                print("3. Modify PyFlex source code to fix env_idx 35 compatibility")
                # Restore signal handler
                signal.signal(signal.SIGSEGV, old_handler)
                if old_alarm_handler is not None:
                    signal.signal(signal.SIGALRM, old_alarm_handler)
            raise
        finally:
            if self.obj == "granular":
                # Restore original signal handler
                try:
                    signal.signal(signal.SIGSEGV, old_handler)
                    if old_alarm_handler is not None:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
                        signal.signal(signal.SIGALRM, old_alarm_handler)
                except:
                    pass

    def get_property_params(self):
        assert self.property_params is not None
        return self.property_params
