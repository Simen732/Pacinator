@echo off
REM Train with WIN-FOCUSED reward function
REM Goal: Teach agent to actually WIN, not just eat pellets
REM 
REM Changes from previous training:
REM - Simplified reward: game score + huge win bonus + death penalty
REM - Death always negative (no progress bonus exploit)
REM - Win bonus: +5000 (massive incentive)
REM - Death penalty: -500 (heavy discouragement)
REM
REM Expected: Agent will take longer to learn, but will learn to SURVIVE and WIN

echo ========================================
echo WIN-FOCUSED TRAINING
echo ========================================
echo Map: mediumClassic
echo Episodes: 20000 (more episodes needed for survival strategies)
echo Reward: Simplified - focus on winning
echo.
echo New reward system:
echo - Game score changes (pellets = +10, etc)
echo - Death: -500 (always bad!)
echo - Win: +5000 (HUGE bonus!)
echo - Step penalty: -1 (encourages efficiency)
echo.
pause

python pacman.py -p PacmanDQN -n 20000 -x 20000 -l test -q --frameTime 0

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo Check logs/ for win rate progress
echo Check saves/ for final checkpoint
echo.
echo To test: Set load_file to the FINAL checkpoint
echo          Set eps=0, eps_final=0
echo          Run: python pacman.py -p PacmanDQN -n 10 -x 0 -l mediumClassic
pause
