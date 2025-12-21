@echo off
SET CONTAINER_NAME=stream-project-spark
SET SCRIPT_PATH=/app/spark/run_dual_sink.sh

echo ======================================================================
echo RUNNING SPARK STREAMING DUAL SINK (ES + MinIO)
echo ======================================================================

echo [1] Checking if container '%CONTAINER_NAME%' is running...
docker inspect -f "{{.State.Running}}" %CONTAINER_NAME% >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Container '%CONTAINER_NAME%' is not running.
    echo Please ensure all Docker services are up.
    echo Run: docker-compose up -d --build
    GOTO :EOF
)
echo Container '%CONTAINER_NAME%' is running.

echo.
echo [2] Executing Spark dual sink streaming job...
docker exec %CONTAINER_NAME% /bin/bash %SCRIPT_PATH%

echo.
echo ======================================================================
echo To check logs: docker logs -f %CONTAINER_NAME%
echo ======================================================================
pause

