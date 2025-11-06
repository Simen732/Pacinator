@echo off
REM Phase 3: Train on test.lay (20x22 map - final map)
REM Time: 25-35 minutes on RTX 3060
REM IMPORTANT: Update load_file in pacmanDQN_Agents.py with Phase 2 checkpoint!

echo Starting Phase 3: test.lay Training (FINAL PHASE)
echo Episodes: 30000
echo Map: test (full complexity)
echo.
echo MAKE SURE you updated load_file with Phase 2 checkpoint!
pause

python pacman.py -p PacmanDQN -n 30000 -x 30000 -l test -q

echo.
echo ============================================
echo Phase 3 Complete - TRAINING FINISHED!
echo ============================================
echo.
echo Your master-level AI is ready!
echo Check saves/ folder for final checkpoint
echo.
pause
