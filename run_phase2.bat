@echo off
REM Phase 2: Train on mediumClassic (20x11 map)
REM Time: 6-10 minutes on RTX 3060
REM IMPORTANT: Update load_file in pacmanDQN_Agents.py with Phase 1 checkpoint!

echo Starting Phase 2: mediumClassic Training
echo Episodes: 10000
echo Map: mediumClassic (medium complexity)
echo.
echo MAKE SURE you updated load_file with Phase 1 checkpoint!
pause

python pacman.py -p PacmanDQN -n 10000 -x 10000 -l mediumClassic -q

echo.
echo Phase 2 Complete!
echo Check saves/ folder for checkpoint
pause
