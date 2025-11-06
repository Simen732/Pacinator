@echo off
REM Watch your trained AI play with graphics!
REM IMPORTANT: Update load_file in pacmanDQN_Agents.py with final checkpoint first!

echo ============================================
echo Pac-inator - AI Gameplay Demo
echo ============================================
echo.
echo MAKE SURE load_file points to your final trained model!
echo.
pause

echo Starting 10 games with graphics...
python pacman.py -p PacmanDQN -n 10 -l test

echo.
echo Demo complete!
pause
