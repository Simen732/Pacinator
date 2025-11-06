@echo off
REM Phase 1: Train on testClassic (5x10 map)
REM Time: 3-5 minutes on RTX 3060
REM Make sure load_file is None in pacmanDQN_Agents.py

echo Starting Phase 1: testClassic Training
echo Episodes: 5000
echo Map: testClassic (smallest)
echo.

python pacman.py -p PacmanDQN -n 5000 -x 5000 -l testClassic -q

echo.
echo Phase 1 Complete!
echo Check saves/ folder for checkpoint
pause
