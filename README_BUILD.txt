GYMPULSE EXE BUILD GUIDE
========================

Goal
----
Create a transferable Windows WebUI folder:
  dist\GymPulseWebUI\

The target PC does NOT need Python, pip, pandas, numpy, or MindSpore installed.
You must transfer the whole dist\GymPulseEngine folder, not only GymPulseEngine.exe.

Runtime files on target PC
--------------------------
Required transfer contents:
  GymPulseWebUI.exe
  _internal\                 <-- PyInstaller dependencies and MindSpore files
  field_dispatch_dataa.csv   <-- editable input CSV beside the exe

Double-click GymPulseWebUI.exe. It starts a private local server and opens the
dashboard in the default browser. The dashboard can download JSON and CSV results.

Build PC requirements
---------------------
Install Python 3.11 64-bit.
Do not use Python 3.14 for this project.

Then open PowerShell in this folder and run:

  powershell -ExecutionPolicy Bypass -File .\build_exe.ps1

Run test after build
--------------------
From this folder:

  .\dist\GymPulseWebUI\GymPulseWebUI.exe

Why this build package fixes the old errors
-------------------------------------------
1. Forces MindSpore to CPU/device 0.
   This avoids "Create device context failed, target device:1 is available".

2. Uses PyInstaller-safe paths.
   CKPT and WebUI templates are read from _internal\pangu_dispatcher_core.
   CSV is read from beside the exe.

3. Uses onedir instead of onefile.
   MindSpore has native/C++ runtime components, so onedir is safer and easier to debug.

Do not include in production exe
--------------------------------
reference_only\test_accuracy.py uses a different 7-feature model shape and older column names.
It is kept only as a reference.
