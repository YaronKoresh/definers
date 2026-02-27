@echo off

if /I NOT "%~dp0"=="%temp%\" (
    copy /y "%~f0" "%temp%\%~nx0" >nul
    start "" "%temp%\%~nx0" %~dp0
    exit /b
)

cd /d "%~1"

:FindGitRoot
if exist ".git" goto :FoundGitRoot
if "%CD%\"=="%~d1\" (
    echo Error: Could not find a .git folder in this directory tree.
    pause
    exit /b
)

cd ..
goto FindGitRoot

:FoundGitRoot
setlocal EnableDelayedExpansion
title Git Manager
color 0B

:MainMenu
cls
echo.
echo ===========================================================
echo                G I T   M A N A G E R
echo ===========================================================
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

if defined CURRENT_BRANCH (
    echo Current Branch: !CURRENT_BRANCH!
) else (
    echo Status: Not inside a Git repository
)

echo.
echo -----------------------------------------------------------
echo  SELECT AN OPTION
echo -----------------------------------------------------------
echo.
echo  [1]  Quick Actions - Common tasks simplified
echo  [2]  First Time Setup
echo  [3]  Branches - Manage separate work areas
echo  [4]  Saving Changes - Take snapshots of your work
echo  [5]  Sync - Upload or download from the cloud
echo  [6]  History - View project details and logs
echo  [7]  Merging - Combine work from different branches
echo  [8]  Undo - Fix mistakes or revert changes
echo  [9]  Versions - Manage tags and releases
echo [10]  Subprojects - Manage nested repositories
echo [11]  Advanced - Power user tools
echo.
echo  [0]  Exit
echo.
echo -----------------------------------------------------------
set "CAT="
set /p "CAT= Enter a number 0-11: "

if "!CAT!"=="1" goto CatQuick
if "!CAT!"=="2" goto CatStart
if "!CAT!"=="3" goto CatBranch
if "!CAT!"=="4" goto CatChanges
if "!CAT!"=="5" goto CatRemote
if "!CAT!"=="6" goto CatHistory
if "!CAT!"=="7" goto CatMerge
if "!CAT!"=="8" goto CatUndo
if "!CAT!"=="9" goto CatTags
if "!CAT!"=="10" goto CatSubmodules
if "!CAT!"=="11" goto CatAdvanced
if "!CAT!"=="0" goto ExitScript

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto MainMenu

:CatQuick
cls
echo.
echo ===========================================================
echo                 Q U I C K   A C T I O N S
echo ===========================================================
echo.
echo  These tasks combine multiple steps into one easy action.
echo.
echo  [1]  Sync - Save work and upload to the cloud
echo  [2]  Update - Get the latest changes from the team
echo  [3]  New Task - Start working on a new branch
echo  [4]  Finish Task - Merge changes and clean up
echo  [5]  Status - See what has been changed
echo  [6]  Reset - Undo everything since the last save
echo  [7]  Identity - Set up your name and email
echo  [8]  Snapshot - Save work locally without uploading
echo  [9]  Release - Create a version and upload it
echo [10]  Download - Get a project for the first time
echo [11]  Undo - Revert a mistake in your last save
echo [12]  Split - Break a big change into smaller saves
echo [13]  Request Review - Create a pull request
echo [14]  Test Review - Download and test a pull request
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "QCH="
set /p "QCH= Select an option 0-14: "

if "!QCH!"=="1" goto DoQuickSaveUpload
if "!QCH!"=="2" goto DoQuickGetLatest
if "!QCH!"=="3" goto DoQuickNewFeature
if "!QCH!"=="4" goto DoQuickFinishFeature
if "!QCH!"=="5" goto DoQuickWhatChanged
if "!QCH!"=="6" goto DoQuickUndoAll
if "!QCH!"=="7" goto DoQuickIdentity
if "!QCH!"=="8" goto DoQuickSaveLocal
if "!QCH!"=="9" goto DoQuickRelease
if "!QCH!"=="10" goto DoQuickDownload
if "!QCH!"=="11" goto DoQuickUndoCommit
if "!QCH!"=="12" goto DoQuickSplitCommit
if "!QCH!"=="13" goto DoQuickCreatePR
if "!QCH!"=="14" goto DoQuickTestPR
if "!QCH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatQuick

:DoQuickSaveUpload
cls
echo.
echo ===========================================================
echo            S A V E   A N D   U P L O A D
echo ===========================================================
echo.
echo  This will mark all changes, create a save point,
echo  get latest team updates, and upload your work.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

echo  Current Branch: !CURRENT_BRANCH!
echo.
echo  Your current changes:
call git status --short
echo.

set "QSAVE_MSG="
set /p "QSAVE_MSG= Describe what you changed: "

if "!QSAVE_MSG!"=="" (
    echo.
    echo Error: You must provide a description to save your work.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 4 - Marking all changes...
call git add -A

echo  Step 2 of 4 - Creating a save point...
call git commit -m "!QSAVE_MSG!"

echo  Step 3 of 4 - Getting latest updates from the cloud...
echo  This ensures your work is compatible with the team.
call git pull origin "!CURRENT_BRANCH!" --no-rebase 2>nul

if errorlevel 1 (
    echo.
    echo  Conflicts detected. Please resolve them before uploading.
    call :ResolveConflicts
)

echo  Step 4 of 4 - Uploading your work...
call git push origin "!CURRENT_BRANCH!" -u

if errorlevel 1 (
    echo.
    echo  Upload failed.
    echo  This usually happens if someone else pushed changes
    echo  while you were working. Try the Update action first.
) else (
    echo.
    echo  Done. Your work is saved and successfully uploaded.
)

echo.
pause
goto CatQuick

:DoQuickGetLatest
cls
echo.
echo ===========================================================
echo            G E T   L A T E S T   U P D A T E S
echo ===========================================================
echo.
echo  This will check for team updates and download them
echo  into your current work area.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

if not defined CURRENT_BRANCH (
    echo Error: You are not inside a Git project.
    pause
    goto CatQuick
)

echo  Current Branch: !CURRENT_BRANCH!
echo.

echo  Step 1 of 2 - Checking for updates...
call git fetch origin

echo  Step 2 of 2 - Downloading updates...
echo  Checking if your local files are ready to be updated...

:: Check if there are local changes that might block a pull
set "HAS_CHANGES="
for /f "delims=" %%i in ('git status --porcelain') do set "HAS_CHANGES=1"

if defined HAS_CHANGES (
    echo.
    echo  Notice: You have unsaved changes.
    echo  Git may refuse to update if these changes overlap
    echo  with the team updates.
    echo.
)

call git pull origin "!CURRENT_BRANCH!" --no-rebase

if errorlevel 1 (
    echo.
    echo  Update paused - Conflicts detected.
    echo  The team changed the same lines you are working on.
    call :ResolveConflicts
)

echo.
echo  Done. Your project is now up to date.
echo.
pause
goto CatQuick

:DoQuickNewFeature
cls
echo.
echo ===========================================================
echo            S T A R T   A   N E W   T A S K
echo ===========================================================
echo.
echo  This creates a separate work area for a new feature.
echo  This keeps your main project code safe and clean.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

echo  You are currently on: !CURRENT_BRANCH!
echo.
echo  Tip: It is best to start new tasks from the main branch.
set "GO_MAIN="
set /p "GO_MAIN= Switch to main branch first? Y or N: "

if /I "!GO_MAIN!"=="Y" (
    echo.
    echo  Moving to main branch...
    call git checkout main 2>nul || call git checkout master 2>nul
    call git pull origin 2>nul
)

echo.
set "QFEAT_NAME="
set /p "QFEAT_NAME= Enter a short name for your task: "

:: Clean up spaces just in case the user included them
set "QFEAT_NAME=!QFEAT_NAME: =_!"

if "!QFEAT_NAME!"=="" (
    echo.
    echo  Error: Task name cannot be empty.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 2 - Creating the work area !QFEAT_NAME!...
call git checkout -b "!QFEAT_NAME!"

echo  Step 2 of 2 - Linking to the cloud...
call git push origin "!QFEAT_NAME!" -u

if errorlevel 1 (
    echo.
    echo  Note: Could not link to cloud. This might be a local-only project.
)

echo.
echo  Done. You are now working on !QFEAT_NAME!.
echo  Your main code is safe. When finished, use the
echo  Finish Task option from the Quick Actions menu.
echo.
pause
goto CatQuick

:DoQuickFinishFeature
cls
echo.
echo ===========================================================
echo            F I N I S H   Y O U R   T A S K
echo ===========================================================
echo.
echo  This will save your work, merge it into the main branch,
echo  upload the combined work, and clean up the old task area.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
set "FEATURE_BRANCH=!CURRENT_BRANCH!"

echo  Current Task: !FEATURE_BRANCH!
echo.

set "QFIN_TARGET=main"
set /p "QFIN_TARGET= Which branch receives the work? Enter for main: "

:: Ensure we aren't trying to merge a branch into itself
if "!FEATURE_BRANCH!"=="!QFIN_TARGET!" (
    echo.
    echo Error: Cannot merge !SOURCE_BR! into itself.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 8 - Saving any remaining work on !FEATURE_BRANCH!...
call git add -A
:: Only commit if there are actually changes to save
call git diff --cached --quiet || call git commit -m "Final changes for !FEATURE_BRANCH!"

echo  Step 2 of 8 - Moving to !QFIN_TARGET!...
call git checkout "!QFIN_TARGET!"
if errorlevel 1 (
    echo Error: Could not switch to !QFIN_TARGET!.
    pause
    goto CatQuick
)

echo  Step 3 of 8 - Searching for unsaved changes in the local !QFIN_TARGET! branch...

call git add -A
call git diff --cached --quiet || (
    set "BR_MSG="
    set /p "BR_MSG= Describe what you changed in !QFIN_TARGET!: "
    call git commit -m "!BR_MSG!"
)

echo  Step 4 of 8 - Getting latest updates from the cloud !QFIN_TARGET!...
call git pull origin "!QFIN_TARGET!" --no-rebase

if errorlevel 1 (
    echo.
    echo Pull failed - Overlaps found.
    call :ResolveConflicts
)

echo  Step 5 of 8 - Combining work from !FEATURE_BRANCH!...
call git merge --no-ff "!FEATURE_BRANCH!"

if errorlevel 1 (
    echo.
    echo Merge paused - Overlaps found.
    call :ResolveConflicts
)

echo  Step 6 of 8 - Uploading combined work to the cloud...
call git push origin "!QFIN_TARGET!"

echo  Step 7 of 8 - Updating local records of the cloud...
call git fetch origin --prune
call git fetch origin !QFIN_TARGET!:!QFIN_TARGET!

echo  Step 8 of 8 - Cleaning up...
echo.
echo  Now that the work is combined, the task branch can be removed.
set "QFIN_CLEANUP="
set /p "QFIN_CLEANUP= Delete the task branch !FEATURE_BRANCH!? Y or N: "

if /I "!QFIN_CLEANUP!"=="Y" (

    echo Removing cloud !FEATURE_BRANCH! branch...
    call git push origin --delete "!FEATURE_BRANCH!" 2>nul

    echo Removing local !FEATURE_BRANCH! branch...
    call git branch -d "!FEATURE_BRANCH!"

    echo.
    echo Cleanup complete.
) else (
    echo Task branch kept for now.
)

echo.
echo Done. Your work is now part of !QFIN_TARGET!.
echo.
pause
goto CatQuick

:DoQuickWhatChanged
cls
echo.
echo ===========================================================
echo               W H A T   H A S   C H A N G E D
echo ===========================================================
echo.
echo  Here is a summary of all your current work.
echo.
echo  --- FILE LIST ---
echo.
call git status --short
echo.
echo  --- SUMMARY OF EDITS ---
echo.
call git diff --stat
echo.
echo  Legend for the File List:
echo   M  = Modified - You changed an existing file
echo   A  = Added    - You created a new file
echo   D  = Deleted  - You removed a file
echo   ?? = Untracked- Git does not know about this file yet
echo.
echo  The Summary of Edits shows how many lines were changed.
echo.
pause
goto CatQuick

:DoQuickUndoAll
cls
echo.
echo ===========================================================
echo             W I P E   A L L   C H A N G E S
echo ===========================================================
echo.
echo  This will permanently delete all work you have done
echo  since your last save point.
echo.
echo  Files to be reset or deleted:
call git status --short
echo.

set "QUNDO_CONFIRM="
set /p "QUNDO_CONFIRM= This action is permanent. Erase everything? Y or N: "

if /I "!QUNDO_CONFIRM!"=="Y" (
    echo.
    echo  Cleaning up work area...

    :: Restore tracked files
    call git restore . 2>nul || call git checkout . 2>nul

    :: Remove new untracked files and folders
    call git clean -fd

    echo.
    echo  Done. Your project is back to how it was at your last save.
) else (
    echo.
    echo  Cancelled. Your changes are still safe.
)

echo.
pause
goto CatQuick

:DoQuickIdentity
cls
echo.
echo ===========================================================
echo                S E T   U P   I D E N T I T Y
echo ===========================================================
echo.
echo  Git needs your name and email to label your save points.
echo  This lets your team know who made which changes.
echo.

set "QID_NAME="
set /p "QID_NAME= Enter your name: "
set "QID_EMAIL="
set /p "QID_EMAIL= Enter your email: "

if "!QID_NAME!"=="" (
    echo Error: Name cannot be empty.
    pause
    goto CatQuick
)

echo.
echo  Where should this identity be used?
echo.
echo  [1]  This project only - Best for work/personal mix
echo  [2]  All projects - Computer-wide default
echo.

set "QID_SCOPE="
set /p "QID_SCOPE= Select a number: "

echo.
if "!QID_SCOPE!"=="2" (
    call git config --global user.name "!QID_NAME!"
    call git config --global user.email "!QID_EMAIL!"
    echo Identity set globally for all projects on this computer.
) else (
    call git config user.name "!QID_NAME!"
    call git config user.email "!QID_EMAIL!"
    echo Identity set specifically for this project only.
)

echo.
echo  Done. Your identity is now: !QID_NAME! ^<!QID_EMAIL!^>
echo.
pause
goto CatQuick

:DoQuickSaveLocal
cls
echo.
echo ===========================================================
echo             S A V E   W O R K   L O C A L L Y
echo ===========================================================
echo.
echo  This creates a save point on your computer only.
echo  Your team will not see these changes until you upload.
echo.
echo  Your current changes:
call git status --short
echo.

set "QLOCAL_MSG="
set /p "QLOCAL_MSG= Describe what you changed: "

if "!QLOCAL_MSG!"=="" (
    echo.
    echo Error: You must provide a description to save your work.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 2 - Marking all changes...
call git add -A

echo  Step 2 of 2 - Creating local save point...
call git commit -m "!QLOCAL_MSG!"

echo.
echo  Done. Your work is saved on this computer.
echo  Remember to use the Upload option later to share it.
echo.
pause
goto CatQuick

:DoQuickRelease
cls
echo.
echo ===========================================================
echo             C R E A T E   A   R E L E A S E
echo ===========================================================
echo.
echo  This marks the current state with a version number
echo  and uploads that mark to the cloud.
echo.

set "QREL_VER="
set /p "QREL_VER= Version name - like v1.0.0: "
set "QREL_MSG="
set /p "QREL_MSG= Describe this release: "

if "!QREL_VER!"=="" (
    echo Error: Version name is required.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 3 - Syncing code before tagging...
for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
call git push origin "!CURRENT_BRANCH!"

echo  Step 2 of 3 - Creating version tag...
call git tag -a "!QREL_VER!" -m "!QREL_MSG!"

echo  Step 3 of 3 - Uploading tag to the cloud...
call git push origin "!QREL_VER!"

if errorlevel 1 (
    echo.
    echo  Warning: The version was created but could not be uploaded.
    echo  Check your internet connection and try Upload One in Tags.
) else (
    echo.
    echo  Done. Version !QREL_VER! is created and uploaded.
)

echo.
pause
goto CatQuick

:DoQuickDownload
cls
echo.
echo ===========================================================
echo             D O W N L O A D   P R O J E C T
echo ===========================================================
echo.
echo  This will download a complete copy of a project
echo  from a cloud link (URL) to your computer.
echo.

set "QDLURL="
set /p "QDLURL= Paste the project URL: "

if "!QDLURL!"=="" (
    echo Error: No URL provided.
    pause
    goto CatQuick
)

set "QDLDIR="
set /p "QDLDIR= Folder name - Enter for default: "

echo.
echo  Downloading... this may take a moment.
echo.

:: We use --recursive to ensure subprojects are downloaded too
if "!QDLDIR!"=="" (
    call git clone --recursive "!QDLURL!"
) else (
    call git clone --recursive "!QDLURL!" "!QDLDIR!"
)

if errorlevel 1 (
    echo.
    echo  Download failed.
    echo  Check the URL and your internet connection.
) else (
    echo.
    echo  Done. The project has been downloaded successfully.
    if not "!QDLDIR!"=="" (
        echo  It is located in the folder: !QDLDIR!
    )
)

echo.
pause
goto CatQuick

:DoQuickUndoCommit
cls
echo.
echo ===========================================================
echo               U N D O   A   S A V E   P O I N T
echo ===========================================================
echo.
echo  Recent save points:
call git log -5 --oneline
echo.

set "QUC_COMMIT=HEAD"
set /p "QUC_COMMIT= Which save point? [Press Enter for the most recent]: "

echo.
echo  How do you want to undo this?
echo.
echo  [1]  Soft Undo - Keep my work but allow me to redo the save
echo  [2]  Hard Erase - Delete the save and all the work inside it
echo  [3]  File Only - Remove just one file from that save point
echo.
set "QUC_CH="
set /p "QUC_CH= Select: "

:: Sync Warning
echo.
echo  Note: If you already uploaded this save to the cloud,
echo  you will need to use a Force Push later to sync the team.
echo.

if "!QUC_CH!"=="1" goto DoUndoKeep
if "!QUC_CH!"=="2" goto DoUndoErase
if "!QUC_CH!"=="3" goto DoUndoFile

echo Invalid choice.
pause
goto CatQuick

:DoUndoKeep
echo.
echo  Step 1 of 1 - Undoing the save point...
call git reset --soft !QUC_COMMIT!~1

echo.
echo  Done. The save point is undone. Your work is still here
echo  and marked for saving. You can now:
echo  - Unmark files you do not want in the Saving Changes menu.
echo  - Create a new save point with a better description.
echo.
echo  Current marked files:
call git diff --cached --name-only

call :PromptForcePush
echo.
pause
goto CatQuick

:DoUndoErase
echo.
echo  WARNING: This will permanently erase the save point
echo  from your project history.
echo.
set "QUC_CONFIRM="
set /p "QUC_CONFIRM= Are you sure you want to proceed? Y or N: "

if /I not "!QUC_CONFIRM!"=="Y" (
    echo.
    echo  Cancelled. No changes were made.
    echo.
    pause
    goto CatQuick
)

echo.
echo  Erasing save point from history...
call :DropSpecificCommit "!QUC_COMMIT!"

:: Erasing history requires a hard force-push if it was already uploaded
call :PromptForcePushHard
echo.
pause
goto CatQuick

:DoUndoFile
echo.
echo  Files inside this save point:
call git diff --name-only !QUC_COMMIT!~1 !QUC_COMMIT!
echo.

set "QUC_FILE="
set /p "QUC_FILE= Enter the file path to remove: "

if "!QUC_FILE!"=="" (
    echo Error: No file specified.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 3 - Recovering original description...
for /f "delims=" %%M in ('git log -1 --format^=%%s !QUC_COMMIT!') do set "QUC_MSG=%%M"

echo  Step 2 of 3 - Unmarking the file !QUC_FILE!...
call git reset --soft !QUC_COMMIT!~1
call git restore --staged "!QUC_FILE!" 2>nul || call git reset HEAD "!QUC_FILE!" 2>nul

echo  Step 3 of 3 - Re-saving remaining work...
call git commit -m "!QSAVE_MSG!"

echo.
echo  Done. !QUC_FILE! has been removed from the save point.
echo  The file still exists on your computer, but it is no
echo  longer included in that specific save.
echo.

call :PromptForcePush
echo.
pause
goto CatQuick

:DoQuickSplitCommit
cls
echo.
echo ===========================================================
echo            B A T C H   S P L I T
echo ===========================================================
echo.

:: Record where we started to show the summary later
for /f "tokens=*" %%a in ('git rev-parse HEAD') do set "START_HASH=%%a"

:SplitLoop
cls
echo  --- UNSAVED FILES ---
echo -----------------------------------------------------------
set "count=0"
for /f "tokens=2" %%f in ('git status --porcelain') do (
    set /a count+=1
    set "file!count!=%%f"
    echo  [!count!] %%f
)
echo -----------------------------------------------------------

if !count!==0 (
    echo.
    echo  [!] No more files to save.
    goto SplitSummary
)

echo.
echo  Enter numbers (e.g., 1, 2, 5-8 or 1 3 4)
echo  'ALL' to grab everything, 'DONE' to finish session.
echo.

set "RAW_PICK="
set /p "RAW_PICK= Selection: "

if /I "!RAW_PICK!"=="DONE" goto SplitSummary
if /I "!RAW_PICK!"=="ALL" (
    call git add -A
    goto SplitDoCommit
)

set "CLEAN_PICK=!RAW_PICK:,= !"
for %%a in (!CLEAN_PICK!) do (
    set "item=%%a"
    echo !item! | findstr "-" >nul
    if !errorlevel!==0 (
        for /f "tokens=1,2 delims=-" %%i in ("!item!") do (
            for /L %%k in (%%i,1,%%j) do call git add "!file%%k!"
        )
    ) else (
        :: Handle single number
        call git add "!file!item!!"
    )
)

:SplitDoCommit
echo.
set "QS_MSG="
set /p "QS_MSG= Description for this group: "
if "!QS_MSG!"=="" set "QS_MSG=Split commit"

set "QS_BR="
set /p "QS_BR= Branch name (Enter for current): "
if not "!QS_BR!"=="" (
    call git checkout -b "!QS_BR!" 2>nul || call git checkout "!QS_BR!"
)

call git commit -m "!QS_MSG!"
echo.
echo  [OK] Group saved.
timeout /t 1 >nul
goto SplitLoop

:SplitSummary
cls
echo.
echo ===========================================================
echo             P U S H   S E L E C T I O N
echo ===========================================================
echo.
echo  New save points created:
echo.
set "new_count=0"
:: Get current branch name
for /f "abbrev-ref HEAD" %%b in ('git rev-parse --abbrev-ref HEAD') do set "CURRENT_BR=%%b"

for /f "tokens=1,*" %%a in ('git log !START_HASH!..HEAD --oneline') do (
    set /a new_count+=1
    set "hash!new_count!=%%a"
    echo  [!new_count!] %%a %%b
)

if !new_count!==0 (
    echo  No new commits found to push.
    pause
    goto CatQuick
)

echo.
echo -----------------------------------------------------------
echo  Which ones to PUSH to !CURRENT_BR!?
echo  (e.g., 1, 2-3, ALL, or NONE)
echo -----------------------------------------------------------

set "PUSH_RAW="
set /p "PUSH_RAW= Push selection: "

if /I "!PUSH_RAW!"=="NONE" goto CatQuick
if /I "!PUSH_RAW!"=="ALL" (
    call git push origin HEAD
    goto CatQuick
)

set "CLEAN_PUSH=!PUSH_RAW:,= !"
for %%a in (!CLEAN_PUSH!) do (
    set "pitem=%%a"
    echo !pitem! | findstr "-" >nul
    if !errorlevel!==0 (
        for /f "tokens=1,2 delims=-" %%i in ("!pitem!") do (
            for /L %%k in (%%i,1,%%j) do (
                echo Pushing !hash%%k!...
                call git push origin !hash%%k!:!CURRENT_BR!
            )
        )
    ) else (
        echo Pushing !hash!pitem!!...
        call git push origin !hash!pitem!!:!CURRENT_BR!
    )
)

echo.
echo  Upload process finished.
pause
goto CatQuick

:DoQuickCreatePR
cls
echo.
echo ===========================================================
echo             C R E A T E   P U L L   R E Q U E S T
echo ===========================================================
echo.
echo  A pull request asks the team to review your changes.
echo  This will upload your work and open the review page.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

echo  You are currently on branch: !CURRENT_BRANCH!
echo.

set "QPR_SAVE="
set /p "QPR_SAVE= Save and upload your latest changes first? Y or N: "

if /I "!QPR_SAVE!"=="Y" (
    echo.
    set "QPR_MSG="
    set /p "QPR_MSG= Describe your changes: "
    echo.
    echo  Step 1 of 3 - Marking all changes...
    call git add -A
    echo  Step 2 of 3 - Creating save point...
    call git commit -m "!QPR_MSG!"
    echo  Step 3 of 3 - Uploading to the cloud...
    call git push origin "!CURRENT_BRANCH!" -u
) else (
    echo.
    echo  Ensuring your branch is uploaded to the cloud...
    call git push origin "!CURRENT_BRANCH!" -u
)

echo.
for /f "delims=" %%U in ('git remote get-url origin 2^>nul') do set "QPR_REMOTE_URL=%%U"

set "QPR_REMOTE_URL=!QPR_REMOTE_URL:.git=!"
set "QPR_REMOTE_URL=!QPR_REMOTE_URL:git@github.com:=https://github.com/!"
set "QPR_REMOTE_URL=!QPR_REMOTE_URL:git@gitlab.com:=https://gitlab.com/!"
set "QPR_REMOTE_URL=!QPR_REMOTE_URL:git@bitbucket.org:=https://bitbucket.org/!"

set "QPR_URL=!QPR_REMOTE_URL!/compare/!CURRENT_BRANCH!?expand=1"

echo  Opening the review page in your browser...
start "" "!QPR_URL!"

echo.
echo  Done. If the browser did not open, you can visit:
echo  !QPR_URL!
echo.
pause
goto CatQuick

:DoQuickTestPR
cls
echo.
echo ===========================================================
echo            T E S T   A   P U L L   R E Q U E S T
echo ===========================================================
echo.
echo  This will download a teammate's pull request so you can
echo  test their changes on your computer.
echo.

set "QTPR_NUM="
set /p "QTPR_NUM= Enter the pull request number: "

if "!QTPR_NUM!"=="" (
    echo Error: No number provided.
    pause
    goto CatQuick
)

echo.
echo  Step 1 of 2 - Downloading the pull request...
call git fetch origin pull/!QTPR_NUM!/head:pr-!QTPR_NUM!

if errorlevel 1 (
    echo.
    echo  Could not download PR !QTPR_NUM!.
    echo  Check that the number is correct and that the
    echo  project is hosted on GitHub.
    echo.
    pause
    goto CatQuick
)

echo  Step 2 of 2 - Switching to the PR branch...
call git checkout pr-!QTPR_NUM!

echo.
echo  Success.
echo.
echo  You are now on a local copy of pull request !QTPR_NUM!.
echo  Test the changes now. When you are finished:
echo   1. Switch back to your branch using the Branches menu.
echo   2. You can safely delete the pr-!QTPR_NUM! branch later.
echo.
pause
goto CatQuick

:CatStart
cls
echo.
echo ===========================================================
echo                F I R S T - T I M E   S E T U P
echo ===========================================================
echo.
echo  [1]  New Project - Start a project from scratch
echo  [2]  Download - Copy an existing project to this folder
echo  [3]  Status - Check if this folder is a Git project
echo  [4]  Set Name - Tell Git who you are
echo  [5]  Set Email - Link your work to your email address
echo  [6]  Git Version - Check if Git is installed correctly
echo  [7]  Project Settings - View specific project rules
echo  [8]  Global Settings - View computer-wide rules
echo  [9]  Edit Global Settings - Change a computer-wide rule
echo [10]  Summary - Show a full project overview
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-10: "

if "!CH!"=="1" goto DoInit
if "!CH!"=="2" goto DoClone
if "!CH!"=="3" goto DoStatus
if "!CH!"=="4" goto DoConfigName
if "!CH!"=="5" goto DoConfigEmail
if "!CH!"=="6" goto DoGitVersion
if "!CH!"=="7" goto DoConfigLocal
if "!CH!"=="8" goto DoConfigGlobal
if "!CH!"=="9" goto DoConfigGlobalSet
if "!CH!"=="10" goto DoRepoInfo
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatStart

:DoInit
cls
echo.
echo ===========================================================
echo             S T A R T   N E W   P R O J E C T
echo ===========================================================
echo.
echo  This will turn this folder into a Git project.
echo.

call git init
call git checkout -b main 2>nul

echo.
echo  Success. This folder is now a Git project.
echo  The default branch has been set to main.
echo.
pause
goto CatStart

:DoClone
cls
echo.
echo ===========================================================
echo             C L O N E   R E P O S I T O R Y
echo ===========================================================
echo.
echo  This downloads an existing project from the cloud.
echo.

set "CLONE_URL="
set /p "CLONE_URL= Enter the repository URL: "

if "!CLONE_URL!"=="" (
    echo Error: No URL provided.
    pause
    goto CatStart
)

set "CLONE_DIR="
set /p "CLONE_DIR= Destination folder - Enter for default: "

echo.
echo  Downloading project...
echo.

if "!CLONE_DIR!"=="" (
    call git clone --recursive "!CLONE_URL!"
) else (
    call git clone --recursive "!CLONE_URL!" "!CLONE_DIR!"
)

if errorlevel 1 (
    echo.
    echo  Download failed. Please check the URL and your connection.
) else (
    echo.
    echo  Success. The project has been downloaded.
)

echo.
pause
goto CatStart

:DoStatus
cls
echo.
echo ===========================================================
echo                P R O J E C T   S T A T U S
echo ===========================================================
echo.

:: Check if we are actually in a git folder before running status
call git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo  This folder is not a Git project yet.
    echo  Use the Start New Project option to begin.
) else (
    echo  Current Status:
    echo.
    call git status
)

echo.
pause
goto CatStart

:DoConfigName
cls
echo.
echo ===========================================================
echo               S E T   U S E R   N A M E
echo ===========================================================
echo.
echo  Your name will be used to label every save point you make.
echo.

set "UNAME="
set /p "UNAME= Enter your full name: "

if "!UNAME!"=="" (
    echo.
    echo Error: Name cannot be blank.
    pause
    goto CatStart
)

call git config user.name "!UNAME!"

echo.
echo Success. Name set to: !UNAME!
echo.
pause
goto CatStart

:DoConfigEmail
cls
echo.
echo ===========================================================
echo              S E T   U S E R   E M A I L
echo ===========================================================
echo.
echo  Your email connects your work to your cloud account.
echo.

set "UEMAIL="
set /p "UEMAIL= Enter your email address: "

if "!UEMAIL!"=="" (
    echo.
    echo Error: Email cannot be blank.
    pause
    goto CatStart
)

call git config user.email "!UEMAIL!"

echo.
echo Success. Email set to: !UEMAIL!
echo.
pause
goto CatStart

:DoGitVersion
cls
echo.
echo ===========================================================
echo                G I T   V E R S I O N
echo ===========================================================
echo.
echo  Checking the version of Git installed on this system...
echo.

call git --version

echo.
echo  If you see a version number above, Git is working correctly.
echo.
pause
goto CatStart

:DoConfigLocal
cls
echo.
echo ===========================================================
echo               L O C A L   C O N F I G
echo ===========================================================
echo.
echo  These settings apply only to this specific project.
echo.

call git config --local --list

echo.
pause
goto CatStart

:DoConfigGlobal
cls
echo.
echo ===========================================================
echo              G L O B A L   C O N F I G
echo ===========================================================
echo.
echo  These settings apply to every project on this computer.
echo.

call git config --global --list

echo.
pause
goto CatStart

:DoConfigGlobalSet
cls
echo.
echo ===========================================================
echo           S E T   G L O B A L   C O N F I G
echo ===========================================================
echo.
echo  Warning: Changing global settings affects all your projects.
echo.

set "GCFG_KEY="
set /p "GCFG_KEY= Enter the setting name - like user.name: "
set "GCFG_VAL="
set /p "GCFG_VAL= Enter the new value: "

if "!GCFG_KEY!"=="" (
    echo.
    echo Error: You must specify which setting to change.
    pause
    goto CatStart
)

call git config --global "!GCFG_KEY!" "!GCFG_VAL!"

echo.
echo Success. Global setting updated:
echo !GCFG_KEY! is now set to !GCFG_VAL!
echo.
pause
goto CatStart

:DoRepoInfo
cls
echo.
echo ===========================================================
echo            R E P O S I T O R Y   D E T A I L S
echo ===========================================================
echo.

call git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo Error: This folder is not a Git project.
    pause
    goto CatStart
)

echo  Project Location:
call git rev-parse --show-toplevel
echo.

echo  Active Branch:
call git rev-parse --abbrev-ref HEAD
echo.

echo  Most Recent Save Point:
call git log -1 --oneline
echo.

echo  Cloud Connections - Remotes:
call git remote -v
echo.

echo  Total Save Points on This Branch:
for /f %%C in ('git rev-list --count HEAD 2^>nul') do echo  %%C
echo.

pause
goto CatStart

:CatBranch
cls
echo.
echo ===========================================================
echo                        B R A N C H E S
echo ===========================================================
echo.
echo  A branch is a parallel copy of your project where you can
echo  work safely without affecting the main version.
echo.
echo  [1]  List Branches - Show all local and cloud branches
echo  [2]  New Branch - Create a new work area
echo  [3]  Switch - Move to a different branch
echo  [4]  Rename - Change the name of your current branch
echo  [5]  Delete - Remove a local branch you no longer need
echo  [6]  Fetch Branch - Download a branch from the cloud
echo  [7]  Link Cloud - Connect this branch to its cloud version
echo  [8]  Done - Show branches already combined into this one
echo  [9]  Pending - Show branches with work not yet merged
echo [10]  Cloud Delete - Remove a branch from the cloud
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-10: "

if "!CH!"=="1" goto DoBranchList
if "!CH!"=="2" goto DoBranchCreate
if "!CH!"=="3" goto DoBranchSwitch
if "!CH!"=="4" goto DoBranchRename
if "!CH!"=="5" goto DoBranchDelete
if "!CH!"=="6" goto DoBranchTrack
if "!CH!"=="7" goto DoBranchUpstream
if "!CH!"=="8" goto DoBranchMerged
if "!CH!"=="9" goto DoBranchUnmerged
if "!CH!"=="10" goto DoBranchDeleteRemote
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatBranch

:DoBranchList
cls
echo.
echo ===========================================================
echo               V I E W   A L L   B R A N C H E S
echo ===========================================================
echo.
echo  Local branches on your computer:
echo  (The one with the * is your active work area)
call git branch
echo.
echo  Branches stored in the cloud:
call git branch -r
echo.
pause
goto CatBranch

:DoBranchCreate
cls
echo.
echo ===========================================================
echo             C R E A T E   N E W   B R A N C H
echo ===========================================================
echo.
echo  A branch creates a separate space for new work.
echo.

set "NEW_BR="
set /p "NEW_BR= Enter a name for the new branch: "

if "!NEW_BR!"=="" (
    echo Error: Branch name cannot be empty.
    pause
    goto CatBranch
)

:: Ensure no spaces in the branch name for Git compatibility
set "NEW_BR=!NEW_BR: =_!"

set "SWITCH_BR="
set /p "SWITCH_BR= Switch to this new branch immediately? Y or N: "

echo.
if /I "!SWITCH_BR!"=="Y" (
    call git checkout -b "!NEW_BR!"
) else (
    call git branch "!NEW_BR!"
)

echo.
echo  Success. The branch !NEW_BR! has been created.
echo.
pause
goto CatBranch

:DoBranchSwitch
cls
echo.
echo ===========================================================
echo               S W I T C H   B R A N C H
echo ===========================================================
echo.
echo  Switching moves your work area to a different branch.
echo.
echo  Available branches:
call git branch
echo.

set "SW_BR="
set /p "SW_BR= Enter the branch name to move to: "

if "!SW_BR!"=="" (
    echo Error: No branch name entered.
    pause
    goto CatBranch
)

echo.
echo  Checking for unsaved work before switching...
call git checkout "!SW_BR!"

if errorlevel 1 (
    echo.
    echo  Switch failed.
    echo  You likely have unsaved changes that would be
    echo  overwritten. Save or Undo your work first.
)

echo.
pause
goto CatBranch

:DoBranchRename
cls
echo.
echo ===========================================================
echo               R E N A M E   B R A N C H
echo ===========================================================
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "OLD_BR=%%I"

echo  Current name: !OLD_BR!
echo.
set "REN_BR="
set /p "REN_BR= Enter the new name for this branch: "

if "!REN_BR!"=="" (
    echo Error: New name cannot be empty.
    pause
    goto CatBranch
)

:: Fix spaces in the new name automatically
set "REN_BR=!REN_BR: =_!"

echo.
echo  Step 1 of 3 - Renaming branch locally...
call git branch -m "!REN_BR!"
set "CURRENT_BRANCH=!REN_BR!"

echo  Step 2 of 3 - Removing the old name from the cloud...
call git push origin --delete "!OLD_BR!" 2>nul

echo  Step 3 of 3 - Uploading the new name to the cloud...
call git push origin -u "!CURRENT_BRANCH!"

echo.
echo  Success. The branch is now named !CURRENT_BRANCH!
echo  on both your computer and the cloud server.
echo.
pause
goto CatBranch

:DoBranchDelete
cls
echo.
echo ===========================================================
echo               D E L E T E   A   B R A N C H
echo ===========================================================
echo.
echo  Note: You cannot delete the branch you are currently on.
echo.
echo  Available local branches:
call git branch
echo.

set "DEL_BR="
set /p "DEL_BR= Enter the branch name to delete: "

if "!DEL_BR!"=="" (
    echo Error: No branch name entered.
    pause
    goto CatBranch
)

set "FORCE_DEL="
set /p "FORCE_DEL= Force delete unmerged work? Y or N: "

echo.
if /I "!FORCE_DEL!"=="Y" (
    call git branch -D "!DEL_BR!"
) else (
    call git branch -d "!DEL_BR!"
)

if errorlevel 1 (
    echo.
    echo  The branch was not deleted. This usually happens if
    echo  it has work that hasn't been merged into main yet.
    pause
    goto CatBranch
)

echo.
set "REMOTE_DEL="
set /p "REMOTE_DEL= Also delete this branch from the cloud? Y or N: "

if /I "!REMOTE_DEL!"=="Y" (
    echo.
    echo  Removing from cloud...
    call git push origin --delete "!DEL_BR!" 2>nul
)

echo.
echo  Done. Cleanup complete.
echo.
pause
goto CatBranch

:DoBranchTrack
cls
echo.
echo ===========================================================
echo           T R A C K   C L O U D   B R A N C H
echo ===========================================================
echo.
echo  This creates a local copy of a branch found on the cloud.
echo.
echo  Remote branches available:
call git branch -r
echo.

set "TRACK_BR="
set /p "TRACK_BR= Enter the cloud branch name - example: origin/task-1: "

if "!TRACK_BR!"=="" (
    echo Error: No branch name entered.
    pause
    goto CatBranch
)

echo.
echo  Connecting to cloud branch and switching...
call git checkout --track "!TRACK_BR!"

if errorlevel 1 (
    echo.
    echo  Failed to track branch.
    echo  Make sure you typed the name exactly as shown above.
)

echo.
pause
goto CatBranch

:DoBranchUpstream
cls
echo.
echo ===========================================================
echo            L I N K   B R A N C H   T O   C L O U D
echo ===========================================================
echo.
echo  This links your local branch to a version on the cloud.
echo  This is required for the Sync and Update actions to work.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

set "DEFAULT_UPS=origin/!CURRENT_BRANCH!"
echo  Current Branch: !CURRENT_BRANCH!
echo.

set "UPS_BR="
set /p "UPS_BR= Enter cloud branch to link to - Enter for !DEFAULT_UPS!: "

if "!UPS_BR!"=="" set "UPS_BR=!DEFAULT_UPS!"

echo.
echo  Linking to !UPS_BR!...
call git branch --set-upstream-to="!UPS_BR!"

if errorlevel 1 (
    echo.
    echo  Link failed. Make sure the branch exists on the cloud.
    echo  You may need to use Upload Work first to create it there.
) else (
    echo.
    echo  Success. This branch is now linked to !UPS_BR!.
)

echo.
pause
goto CatBranch

:DoBranchMerged
cls
echo.
echo ===========================================================
echo            S A F E   T O   D E L E T E
echo ===========================================================
echo.
echo  These branches have already been combined into your
echo  current work area and are safe to remove.
echo.

call git branch --merged

echo.
pause
goto CatBranch

:DoBranchUnmerged
cls
echo.
echo ===========================================================
echo            A C T I V E   W O R K
echo ===========================================================
echo.
echo  These branches contain changes that have NOT been
echo  combined into your current branch yet.
echo.

call git branch --no-merged

echo.
echo  Be careful: Deleting these will result in lost work.
echo.
pause
goto CatBranch

:DoBranchDeleteRemote
cls
echo.
echo ===========================================================
echo            D E L E T E   C L O U D   B R A N C H
echo ===========================================================
echo.
echo  This removes a branch from the cloud server.
echo  Teammates will no longer see this branch.
echo.
echo  Cloud branches:
call git branch -r
echo.

set "RDBR_NAME="
set /p "RDBR_NAME= Enter branch name to delete: "

if "!RDBR_NAME!"=="" (
    echo Error: No branch name provided.
    pause
    goto CatBranch
)

:: If user pasted 'origin/branch-name', we strip 'origin/' for the command
set "RDBR_NAME=!RDBR_NAME:origin/=!"

set "RDBR_REMOTE=origin"
set /p "RDBR_REMOTE= Remote name - Enter for origin: "

echo.
set "RDBR_CONFIRM="
set /p "RDBR_CONFIRM= Delete !RDBR_NAME! from !RDBR_REMOTE!? Y or N: "

if /I "!RDBR_CONFIRM!"=="Y" (
    echo.
    echo  Removing branch from the cloud...
    call git push "!RDBR_REMOTE!" --delete "!RDBR_NAME!"

    if errorlevel 1 (
        echo.
        echo  Failed to delete. The branch may already be gone
        echo  or you might not have permission to delete it.
    ) else (
        echo.
        echo  Success. The cloud branch has been removed.
    )
) else (
    echo.
    echo  Cancelled. No changes were made.
)

echo.
pause
goto CatBranch

:CatChanges
cls
echo.
echo ===========================================================
echo                 S A V I N G   C H A N G E S
echo ===========================================================
echo.
echo  Marking = Choosing files to include in your next save.
echo  Saving = Creating a permanent snapshot of marked files.
echo  Set Aside = Temporarily hiding work to clean your folder.
echo.
echo  [1]  Mark All - Prepare all changed files for saving
echo  [2]  Mark File - Choose a specific file to save
echo  [3]  Unmark - Remove a file from the next save
echo  [4]  Save Marked - Create a save point for marked files
echo  [5]  Quick Save - Mark everything and save at once
echo  [6]  Set Aside - Hide your current changes for later
echo  [7]  Restore - Bring back your set-aside changes
echo  [8]  List Hidden - Show all set-aside work
echo  [9]  Discard File - Delete changes in a specific file
echo [10]  Mark Parts - Choose specific lines to save
echo [11]  Preview Hidden - See what is inside a set-aside entry
echo [12]  Delete Hidden - Remove one set-aside entry
echo [13]  Clear Hidden - Delete ALL set-aside entries
echo [14]  Compare - See what changed inside a file
echo [15]  Reset All - Throw away ALL unsaved changes
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-15: "

if "!CH!"=="1" goto DoStageAll
if "!CH!"=="2" goto DoStageFile
if "!CH!"=="3" goto DoUnstage
if "!CH!"=="4" goto DoCommit
if "!CH!"=="5" goto DoQuickCommit
if "!CH!"=="6" goto DoStash
if "!CH!"=="7" goto DoStashPop
if "!CH!"=="8" goto DoStashList
if "!CH!"=="9" goto DoDiscardFile
if "!CH!"=="10" goto DoStagePatch
if "!CH!"=="11" goto DoStashShow
if "!CH!"=="12" goto DoStashDrop
if "!CH!"=="13" goto DoStashClear
if "!CH!"=="14" goto DoDiffFile
if "!CH!"=="15" goto DoDiscardAll
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatChanges

:DoStageAll
cls
echo.
echo ===========================================================
echo             M A R K   A L L   C H A N G E S
echo ===========================================================
echo.
echo  This marks every change you have made as ready to be saved.
echo.

call git add -A

echo  Success. All changes are now marked and ready for a save point.
echo.
pause
goto CatChanges

:DoStageFile
cls
echo.
echo ===========================================================
echo            M A R K   S P E C I F I C   F I L E
echo ===========================================================
echo.
echo  Modified files currently on your computer:
call git status --short
echo.

set "STAGE_F="
set /p "STAGE_F= Enter the file path to mark: "

if "!STAGE_F!"=="" (
    echo.
    echo Error: No file specified.
    pause
    goto CatChanges
)

call git add "!STAGE_F!"

echo.
echo  Done. !STAGE_F! is now marked for saving.
echo.
pause
goto CatChanges

:DoUnstage
cls
echo.
echo ===========================================================
echo               U N M A R K   A   F I L E
echo ===========================================================
echo.
echo  Files currently marked for the next save point:
call git diff --cached --name-only
echo.

set "UNSTAGE_F="
set /p "UNSTAGE_F= Enter the file path to unmark: "

if "!UNSTAGE_F!"=="" (
    echo.
    echo Error: No file specified.
    pause
    goto CatChanges
)

echo.
echo  Unmarking !UNSTAGE_F!...
:: Use restore with a fallback to the classic reset command
call git restore --staged "!UNSTAGE_F!" 2>nul || call git reset HEAD "!UNSTAGE_F!" 2>nul

echo.
echo  Done. The file has been unmarked.
echo  Your work is still on your computer, just not in the next save.
echo.
pause
goto CatChanges

:DoCommit
cls
echo.
echo ===========================================================
echo             C R E A T E   S A V E   P O I N T
echo ===========================================================
echo.
echo  This saves all your 'Marked' files into your history.
echo.

set "COMMIT_MSG="
set /p "COMMIT_MSG= Describe what you changed: "

if "!COMMIT_MSG!"=="" (
    echo.
    echo Error: You must provide a description to save your work.
    pause
    goto CatChanges
)

echo.
echo  Saving changes...
call git commit -m "!COMMIT_MSG!"

if errorlevel 1 (
    echo.
    echo  Save failed. Make sure you have marked (staged)
    echo  your files first.
) else (
    echo.
    echo  Success. Your work has been saved locally.
)

echo.
pause
goto CatChanges

:DoQuickCommit
cls
echo.
echo ===========================================================
echo               Q U I C K   S A V E
echo ===========================================================
echo.
echo  This marks ALL current changes and saves them instantly.
echo.

set "QC_MSG="
set /p "QC_MSG= Describe what you changed: "

if "!QC_MSG!"=="" (
    echo.
    echo Error: You must provide a description to save your work.
    pause
    goto CatChanges
)

echo.
echo  Step 1 of 2 - Marking all changes...
call git add -A

echo  Step 2 of 2 - Creating save point...
call git commit -m "!QC_MSG!"

echo.
echo  Done. Your changes are now saved to your local history.
echo.
pause
goto CatChanges

:DoStash
cls
echo.
echo ===========================================================
echo             S H E L V E   W O R K   (STASH)
echo ===========================================================
echo.
echo  This moves your current changes to a temporary shelf.
echo  Your project will return to its last saved state.
echo.

set "STASH_MSG="
set /p "STASH_MSG= Description for this work - Enter to skip: "

echo.
if "!STASH_MSG!"=="" (
    call git stash
) else (
    call git stash push -m "!STASH_MSG!"
)

echo.
echo  Done. Your changes have been moved to the shelf.
echo.
pause
goto CatChanges

:DoStashPop
cls
echo.
echo ===========================================================
echo             R E T R I E V E   W O R K
echo ===========================================================
echo.
echo  This takes work off your shelf and puts it back
echo  into your project.
echo.
echo  Available items on shelf:
call git stash list
echo.

set "STASH_IDX="
set /p "STASH_IDX= Enter number to retrieve - Enter for latest: "

echo.
echo  Retrieving work...
if "!STASH_IDX!"=="" (
    call git stash pop
) else (
    :: We use quotes around the index to prevent Batch parser errors
    call git stash pop "stash@{!STASH_IDX!}"
)

if errorlevel 1 (
    echo.
    echo  Could not retrieve work.
    echo  This usually happens if your current project files
    echo  conflict with the work on the shelf.
)

echo.
pause
goto CatChanges

:DoStashList
cls
echo.
echo ===========================================================
echo                S H E L F   C O N T E N T S
echo ===========================================================
echo.
echo  These are the items currently saved on your temporary shelf:
echo.

call git stash list

echo.
pause
goto CatChanges

:DoDiscardFile
cls
echo.
echo ===========================================================
echo            D I S C A R D   C H A N G E S
echo ===========================================================
echo.
echo  WARNING: This will permanently delete your unsaved work
echo  in a specific file and return it to the last save point.
echo.
echo  Modified files:
call git status --short
echo.

set "DISCARD_F="
set /p "DISCARD_F= Enter the file path to reset: "

if "!DISCARD_F!"=="" (
    echo.
    echo Error: No file specified.
    pause
    goto CatChanges
)

set "CONFIRM_DISCARD="
set /p "CONFIRM_DISCARD= This action CANNOT BE UNDONE. Proceed? Y or N: "

if /I "!CONFIRM_DISCARD!"=="Y" (
    echo.
    echo  Restoring file...
    call git restore "!DISCARD_F!" 2>nul || call git checkout -- "!DISCARD_F!" 2>nul
    echo  Changes discarded successfully.
) else (
    echo.
    echo  Cancelled. Your work is safe.
)

echo.
pause
goto CatChanges

:DoStagePatch
cls
echo.
echo ===========================================================
echo             S E L E C T I V E   M A R K I N G
echo ===========================================================
echo.
echo  This lets you choose which specific lines of code to mark
echo  for saving, even if they are in the same file.
echo.
echo  Quick Guide:
echo   y = Mark this piece
echo   n = Skip this piece
echo   s = Split into smaller pieces
echo   q = Quit and finish
echo.
echo  Starting interactive mode...
echo.

call git add -p

echo.
echo  Interactive marking complete.
echo.
pause
goto CatChanges

:DoStashShow
cls
echo.
echo ===========================================================
echo            I N S P E C T   S H E L V E D   W O R K
echo ===========================================================
echo.
echo  Available items on your shelf:
call git stash list
echo.

set "SSHOW_IDX="
set /p "SSHOW_IDX= Enter number to inspect - Enter for latest: "

echo.
echo  Showing changes...
echo.

if "!SSHOW_IDX!"=="" (
    call git stash show -p
) else (
    call git stash show -p "stash@{!SSHOW_IDX!}"
)

echo.
pause
goto CatChanges

:DoStashDrop
cls
echo.
echo ===========================================================
echo             T H R O W   A W A Y   S H E L F   I T E M
echo ===========================================================
echo.
echo  Available items on your shelf:
call git stash list
echo.

set "SDROP_IDX="
set /p "SDROP_IDX= Enter the number of the item to delete: "

if "!SDROP_IDX!"=="" (
    echo Error: No number provided.
    pause
    goto CatChanges
)

echo.
set "SDROP_CONFIRM="
set /p "SDROP_CONFIRM= This will delete the work PERMANENTLY. Continue? Y or N: "

if /I "!SDROP_CONFIRM!"=="Y" (
    call git stash drop "stash@{!SDROP_IDX!}"
    echo Item deleted from shelf.
) else (
    echo Cancelled.
)

echo.
pause
goto CatChanges

:DoStashClear
cls
echo.
echo ===========================================================
echo                E M P T Y   T H E   S H E L F
echo ===========================================================
echo.
echo  Current items on shelf:
call git stash list
echo.

set "SCLEAR_CONFIRM="
set /p "SCLEAR_CONFIRM= Delete EVERY item on the shelf permanently? Y or N: "

if /I "!SCLEAR_CONFIRM!"=="Y" (
    call git stash clear
    echo.
    echo Success. The shelf is now empty.
) else (
    echo.
    echo Cancelled.
)

echo.
pause
goto CatChanges

:DoDiffFile
cls
echo.
echo ===========================================================
echo             I N S P E C T   C H A N G E S
echo ===========================================================
echo.
echo  Modified files on your computer:
call git status --short
echo.

set "DIFF_F="
set /p "DIFF_F= Enter the file path to inspect: "

if "!DIFF_F!"=="" (
    echo Error: No file specified.
    pause
    goto CatChanges
)

echo.
echo  Which version do you want to see?
echo.
echo  [1]  Unmarked Changes - Work you have not prepared for a save yet
echo  [2]  Marked Changes - Work you have already prepared for a save
echo.

set "DIFF_FCH="
set /p "DIFF_FCH= Select: "

echo.
if "!DIFF_FCH!"=="1" call git diff "!DIFF_F!"
if "!DIFF_FCH!"=="2" call git diff --cached "!DIFF_F!"

echo.
pause
goto CatChanges

:DoDiscardAll
cls
echo.
echo ===========================================================
echo            D I S C A R D   A L L   W O R K
echo ===========================================================
echo.
echo  Current status:
call git status --short
echo.

set "DALL_CONFIRM="
set /p "DALL_CONFIRM= This will permanently delete ALL unsaved work. Continue? Y or N: "

if /I "!DALL_CONFIRM!"=="Y" (
    echo.
    echo  Step 1 of 2 - Restoring modified files...
    call git restore .

    echo  Step 2 of 2 - Removing new untracked files...
    call git clean -fd

    echo.
    echo  Success. All local changes have been wiped.
) else (
    echo.
    echo  Cancelled. Your work is safe.
)

echo.
pause
goto CatChanges

:CatRemote
cls
echo.
echo ===========================================================
echo            U P L O A D   A N D   D O W N L O A D
echo ===========================================================
echo.
echo  Push = Upload your saves to the cloud server.
echo  Pull = Download and apply updates from the team.
echo  Remote = The internet address where your project lives.
echo.
echo  [1]  Upload - Send your saves to the cloud
echo  [2]  Download - Get updates and apply them now
echo  [3]  Check - Look for updates without applying them
echo  [4]  List Servers - Show saved cloud addresses
echo  [5]  Connect - Link to a new cloud server
echo  [6]  Disconnect - Remove a cloud server connection
echo  [7]  Rename - Change the name of a server link
echo  [8]  Update URL - Change a server internet address
echo  [9]  Cleanup - Remove references to deleted branches
echo [10]  Upload All - Send all branches to the cloud
echo [11]  Upload Tags - Send all version marks to the cloud
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-11: "

if "!CH!"=="1" goto DoPush
if "!CH!"=="2" goto DoPull
if "!CH!"=="3" goto DoFetch
if "!CH!"=="4" goto DoRemoteList
if "!CH!"=="5" goto DoRemoteAdd
if "!CH!"=="6" goto DoRemoteRemove
if "!CH!"=="7" goto DoRemoteRename
if "!CH!"=="8" goto DoRemoteSetUrl
if "!CH!"=="9" goto DoRemotePrune
if "!CH!"=="10" goto DoPushAll
if "!CH!"=="11" goto DoPushTags
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatRemote

:DoPush
cls
echo.
echo ===========================================================
echo             U P L O A D   W O R K   (PUSH)
echo ===========================================================
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
echo  Current Branch: !CURRENT_BRANCH!

set "TARGET_BRANCH=!CURRENT_BRANCH!"
set "PUBLISH_FLAG="
set "FORCE_FLAG="

set "INPUT_BRANCH="
set /p "INPUT_BRANCH= Upload to which branch? Enter for !CURRENT_BRANCH!: "
if not "!INPUT_BRANCH!"=="" set "TARGET_BRANCH=!INPUT_BRANCH!"

call git rev-parse --verify --quiet refs/heads/!TARGET_BRANCH! >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Branch !TARGET_BRANCH! does not exist on your computer.
    set "CREATE_BRANCH="
    set /p "CREATE_BRANCH= Create and switch to it now? Y or N: "
    if /I "!CREATE_BRANCH!"=="Y" (
        call git checkout -b "!TARGET_BRANCH!"
        set "DO_PUBLISH="
        set /p "DO_PUBLISH= Share this new branch with the cloud? Y or N: "
        if /I "!DO_PUBLISH!"=="Y" set "PUBLISH_FLAG=-u"
    ) else (
        echo Cancelled.
        pause
        goto CatRemote
    )
) else (
    if not "!TARGET_BRANCH!"=="!CURRENT_BRANCH!" (
        echo  Switching to !TARGET_BRANCH!...
        call git checkout "!TARGET_BRANCH!"
    )
)

echo.
set "FORCE_PUSH="
set /p "FORCE_PUSH= Do you need to overwrite the cloud (Force Push)? Y or N: "

if /I not "!FORCE_PUSH!"=="Y" (
    set "FORCE_FLAG="
    echo Safe upload selected.
    goto SkipForcePush
)

echo.
echo  ####################  DANGER  ####################
echo  This will OVERWRITE the cloud project with your
echo  local version. Any work on the cloud that you
echo  do not have on your computer will be DELETED.
echo  ##################################################
echo.
set "CONFIRM_OVERWRITE="
set /p "CONFIRM_OVERWRITE= Type OVERWRITE to proceed or press Enter to cancel: "
if /I "!CONFIRM_OVERWRITE!"=="OVERWRITE" (
    set "FORCE_FLAG=--force-with-lease"
    echo Cloud overwrite enabled safely.
) else (
    set "FORCE_FLAG="
    echo Safe upload selected.
)

:SkipForcePush
echo.
echo Automated quality checks are important, but they can be skipped when needed.
echo.
set "SKIP_VERIFICATION="
set /p "SKIP_VERIFICATION= Skip automated quality checks? Y or N: "
if /I "!SKIP_VERIFICATION!"=="Y" (
    set "NO_VERIFY=--no-verify"
    echo Quality checks disabled.
) else (
    set "NO_VERIFY="
    echo Using default quality checks.
)

echo.
echo Before sharing, you should bundle your changes into a save point.
echo.
set "WANT_COMMIT="
set /p "WANT_COMMIT= Save your work before uploading? Y or N: "
if /I "!WANT_COMMIT!"=="Y" (
    set "PUSH_MSG="
    set /p "PUSH_MSG= Describe your changes: "
    if "!PUSH_MSG!"=="" set "PUSH_MSG=Quick upload save"
    call git add -A
    call git commit -m "!PUSH_MSG!" !NO_VERIFY!
)

if not defined FORCE_FLAG (
    if not "!PUBLISH_FLAG!"=="-u" (
        echo.
        echo  Checking for cloud updates before uploading...
        call git pull origin "!TARGET_BRANCH!" --no-rebase

        if errorlevel 1 (
            echo.
            echo  -----------------------------------------------------------
            echo  CONFLICT DETECTED: A teammate updated the cloud.
            echo  We must merge their work into yours before you can upload.
            echo  -----------------------------------------------------------
            pause

            :: Launch your Conflict Resolution Tool
            call :ResolveConflicts

            :: Check if the conflict was actually resolved or if they aborted
            call git status --porcelain | findstr "^UU ^AA ^DU ^UD" >nul
            if not errorlevel 1 (
                echo.
                echo  Upload cancelled. Conflicts are still active.
                pause
                goto CatRemote
            )
            echo.
            echo  Conflicts resolved. Proceeding with upload...
        )
    )
)

echo.
echo  Uploading to origin/!TARGET_BRANCH!...
call git push origin "!TARGET_BRANCH!" !FORCE_FLAG! !PUBLISH_FLAG! !NO_VERIFY!

if errorlevel 1 (
    echo.
    echo  Upload failed.
    echo  If this was not a conflict, check your internet connection
    echo  or your permissions for this repository.
) else (
    echo.
    echo  Success. Your work is now on the cloud.
)

echo.
pause
goto CatRemote

:DoPull
cls
echo.
echo ===========================================================
echo             U P D A T E   P R O J E C T   (PULL)
echo ===========================================================
echo.
echo  This downloads the latest changes from the cloud and
echo  combines them with your local work.
echo.

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

set "PULL_BR="
set /p "PULL_BR= Which branch to download? Enter for !CURRENT_BRANCH!: "
if "!PULL_BR!"=="" set "PULL_BR=!CURRENT_BRANCH!"

echo.
echo  Checking for unsaved work before updating...
echo.

:: Check if there are local changes that might block the pull
call git diff --quiet
if errorlevel 1 (
    echo  Warning: You have unsaved changes.
    echo  If the cloud has changed the same files, the update may fail.
    echo.
    set "SAVE_FIRST="
    set /p "SAVE_FIRST= Would you like to create a Save Point first? Y or N: "
    if /I "!SAVE_FIRST!"=="Y" (
        set "PULL_MSG="
        set /p "PULL_MSG= Describe your current work: "
        call git add -A
        call git commit -m "!PULL_MSG!"
    )
)

echo.
echo  Downloading and merging changes from origin/!PULL_BR!...
call git pull origin "!PULL_BR!" --no-rebase

if errorlevel 1 (
    echo.
    echo  Update stopped. There is a conflict between your
    echo  work and the cloud version.
    echo.
    pause
    call :ResolveConflicts
) else (
    echo.
    echo  Success. Your project is now up to date.
)

echo.
pause
goto CatRemote

:DoFetch
cls
echo.
echo ===========================================================
echo             C H E C K   F O R   U P D A T E S
echo ===========================================================
echo.
echo  This asks the cloud if there are any new changes,
echo  but it does NOT change your files yet.
echo.

call git fetch --all

echo.
echo  Check complete. You can now see what others have done
echo  without affecting your current work.
echo.
pause
goto CatRemote

:DoRemoteList
cls
echo.
echo ===========================================================
echo            C L O U D   C O N N E C T I O N S
echo ===========================================================
echo.
echo  These are the cloud servers your project is linked to:
echo.

call git remote -v

echo.
pause
goto CatRemote

:DoRemoteAdd
cls
echo.
echo ===========================================================
echo            A D D   C L O U D   L I N K
echo ===========================================================
echo.
echo  Link your local project to a server like GitHub or GitLab.
echo.

set "REM_NAME="
set /p "REM_NAME= Name this connection - Enter for origin: "
if "!REM_NAME!"=="" set "REM_NAME=origin"

set "REM_URL="
set /p "REM_URL= Paste the Cloud URL here: "

if "!REM_URL!"=="" (
    echo.
    echo Error: You must provide a URL to connect to.
    pause
    goto CatRemote
)

echo.
echo  Connecting to !REM_NAME!...
call git remote add "!REM_NAME!" "!REM_URL!"

if errorlevel 1 (
    echo.
    echo  Failed to add link. Check if the name already exists.
) else (
    echo  Success. Downloading cloud information...
    call git fetch "!REM_NAME!"
    echo.
    echo  The cloud link is ready to use.
)

echo.
pause
goto CatRemote

:DoRemoteRemove
cls
echo.
echo ===========================================================
echo          R E M O V E   C L O U D   L I N K
echo ===========================================================
echo.
echo  This stops your computer from syncing with a specific server.
echo  It will NOT delete any files on the cloud or your computer.
echo.
echo  Active connections:
call git remote -v
echo.

set "REM_DEL="
set /p "REM_DEL= Enter the name of the link to remove: "

if "!REM_DEL!"=="" (
    echo Error: No name entered.
    pause
    goto CatRemote
)

echo.
echo  Removing link !REM_DEL!...
call git remote remove "!REM_DEL!"

echo.
echo  Done. The connection has been removed.
echo.
pause
goto CatRemote

:DoRemoteRename
cls
echo.
echo ===========================================================
echo            R E N A M E   C L O U D   L I N K
echo ===========================================================
echo.
echo  Current cloud connections:
call git remote -v
echo.

set "REM_OLD="
set /p "REM_OLD= Enter the current name - e.g. origin: "
set "REM_NEW="
set /p "REM_NEW= Enter the new name: "

if "!REM_OLD!"=="" (
    echo Error: Current name is required.
    pause
    goto CatRemote
)

echo.
echo  Renaming !REM_OLD! to !REM_NEW!...
call git remote rename "!REM_OLD!" "!REM_NEW!"

if errorlevel 1 (
    echo.
    echo  Rename failed. Make sure the current name is correct.
) else (
    echo  Success. The link has been renamed.
)

echo.
pause
goto CatRemote

:DoRemoteSetUrl
cls
echo.
echo ===========================================================
echo            U P D A T E   C L O U D   U R L
echo ===========================================================
echo.
echo  Use this if your project moved to a new web address.
echo.
echo  Current cloud connections:
call git remote -v
echo.

set "RURL_NAME="
set /p "RURL_NAME= Enter the connection name - e.g. origin: "
set "RURL_URL="
set /p "RURL_URL= Enter the new Cloud URL: "

if "!RURL_URL!"=="" (
    echo Error: New URL cannot be empty.
    pause
    goto CatRemote
)

echo.
echo  Updating address for !RURL_NAME!...
call git remote set-url "!RURL_NAME!" "!RURL_URL!"

if errorlevel 1 (
    echo.
    echo  Update failed. Check the connection name and try again.
) else (
    echo  Success. The URL has been updated.
)

echo.
pause
goto CatRemote

:DoRemotePrune
cls
echo.
echo ===========================================================
echo            C L E A N   S T A L E   L I N K S
echo ===========================================================
echo.
echo  This removes "ghost" branches from your list that were
echo  already deleted from the cloud by your teammates.
echo.

set "PRUNE_REM=origin"
set /p "PRUNE_REM= Which cloud to prune? Enter for origin: "

echo.
echo  Cleaning up stale branches from !PRUNE_REM!...
call git remote prune "!PRUNE_REM!"

echo.
echo  Cleanup complete. Your branch list is now up to date.
echo.
pause
goto CatRemote

:DoPushAll
cls
echo.
echo ===========================================================
echo          U P L O A D   A L L   B R A N C H E S
echo ===========================================================
echo.
echo  This sends every branch on your computer to the cloud.
echo  It is a great way to ensure your entire project is
echo  backed up and synced.
echo.

set "PALL_REM=origin"
set /p "PALL_REM= Remote to push to - Enter for origin: "

echo.
echo  Uploading all branches to !PALL_REM!...
call git push "!PALL_REM!" --all

if errorlevel 1 (
    echo.
    echo  Upload failed. Check your internet connection or
    echo  permissions for this repository.
) else (
    echo.
    echo  Success. All your branches are now on the cloud.
)

echo.
pause
goto CatRemote

:DoPushTags
cls
echo.
echo ===========================================================
echo                U P L O A D   T A G S
echo ===========================================================
echo.
echo  Tags are version labels (like v1.0). This sends your
echo  milestone markers to the cloud so others can see them.
echo.

set "PTAG_REM=origin"
set /p "PTAG_REM= Remote to push to - Enter for origin: "

echo.
echo  Uploading tags to !PTAG_REM!...
call git push "!PTAG_REM!" --tags

if errorlevel 1 (
    echo.
    echo  Upload failed.
) else (
    echo.
    echo  Success. All version tags have been shared.
)

echo.
pause
goto CatRemote

:CatHistory
cls
echo.
echo ===========================================================
echo                H I S T O R Y   A N D   L O G S
echo ===========================================================
echo.
echo  [1]  Full History - Show recent save points with details
echo  [2]  Compact List - Show recent save points in one line
echo  [3]  Compare - See what you have changed right now
echo  [4]  Inspect - Show details of one specific save point
echo  [5]  Search - Find save points by their description
echo  [6]  Track Changes - See who wrote each line in a file
echo  [7]  File History - Show all changes made to one file
echo  [8]  Action Log - Show all recent actions and undoes
echo  [9]  Contributors - See who has worked on this project
echo [10]  Time Travel - View a file as it was in the past
echo [11]  Total Count - Show the total number of save points
echo [12]  Compare Points - See differences between two saves
echo [13]  File List - List every file inside this project
echo [14]  Statistics - Show lines added or removed per save
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-14: "

if "!CH!"=="1" goto DoLog
if "!CH!"=="2" goto DoLogOneline
if "!CH!"=="3" goto DoDiff
if "!CH!"=="4" goto DoShowCommit
if "!CH!"=="5" goto DoSearchCommit
if "!CH!"=="6" goto DoBlame
if "!CH!"=="7" goto DoFileHistory
if "!CH!"=="8" goto DoReflog
if "!CH!"=="9" goto DoShortlog
if "!CH!"=="10" goto DoShowFileAtCommit
if "!CH!"=="11" goto DoCountCommits
if "!CH!"=="12" goto DoDiffCommits
if "!CH!"=="13" goto DoLsFiles
if "!CH!"=="14" goto DoCommitStats
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatHistory

:DoLog
cls
echo.
echo ===========================================================
echo             D E T A I L E D   H I S T O R Y
echo ===========================================================
echo.
echo  This view shows who made each save point, the date,
echo  and the full description.
echo.

set "LOG_N="
set /p "LOG_N= How many save points to show? Enter for 10: "
if "!LOG_N!"=="" set "LOG_N=10"

echo.
echo  Note: Use arrow keys to scroll. Press 'Q' to exit the list.
echo -----------------------------------------------------------
echo.

call git log -!LOG_N! --graph --decorate

echo.
echo -----------------------------------------------------------
echo  Returned to menu.
echo.
pause
goto CatHistory

:DoLogOneline
cls
echo.
echo ===========================================================
echo              C O M P A C T   H I S T O R Y
echo ===========================================================
echo.
echo  This view shows a clean, one-line summary of your
echo  project timeline.
echo.

set "LOG_N2="
set /p "LOG_N2= How many save points to show? Enter for 20: "
if "!LOG_N2!"=="" set "LOG_N2=20"

echo.
echo  Note: Use arrow keys to scroll. Press 'Q' to exit the list.
echo -----------------------------------------------------------
echo.

call git log -!LOG_N2! --oneline --graph --decorate

echo.
echo -----------------------------------------------------------
echo  Returned to menu.
echo.
pause
goto CatHistory

:DoDiff
cls
echo.
echo ===========================================================
echo             C O M P A R E   C H A N G E S
echo ===========================================================
echo.
echo  Choose what you want to compare:
echo.
echo  [1]  Unmarked Changes - Work you haven't prepared for a save yet
echo  [2]  Marked Changes   - Work you have already prepared to save
echo  [3]  Two Branches     - See the differences between two versions
echo.
set "DIFF_CH="
set /p "DIFF_CH= Select: "

if "!DIFF_CH!"=="1" (
    echo.
    echo  Showing changes you have not marked yet...
    echo  Note: Press 'Q' to exit the view.
    echo.
    call git diff
)

if "!DIFF_CH!"=="2" (
    echo.
    echo  Showing changes you have already marked for saving...
    echo  Note: Press 'Q' to exit the view.
    echo.
    call git diff --cached
)

if "!DIFF_CH!"=="3" (
    echo.
    echo  Available branches:
    call git branch
    echo.
    set "DIFF_A="
    set /p "DIFF_A= Enter the name of the first branch: "
    set "DIFF_B="
    set /p "DIFF_B= Enter the name of the second branch: "

    if "!DIFF_A!"=="" goto :DiffError
    if "!DIFF_B!"=="" goto :DiffError

    echo.
    echo  Comparing !DIFF_A! against !DIFF_B!...
    echo  Note: Press 'Q' to exit the view.
    echo.
    call git diff "!DIFF_A!".."!DIFF_B!"
)

echo.
pause
goto CatHistory

:DiffError
echo.
echo  Error: You must provide two branch names to compare.
pause
goto CatHistory

:DoShowCommit
cls
echo.
echo ===========================================================
echo            V I E W   S A V E   P O I N T   (SHOW)
echo ===========================================================
echo.
echo  This shows exactly what files were changed and what code
echo  was added or removed in a specific save point.
echo.

set "SHOW_SHA="
set /p "SHOW_SHA= Enter the ID Number (Hash) of the save: "

if "!SHOW_SHA!"=="" (
    echo.
    echo Error: No ID Number provided.
    pause
    goto CatHistory
)

echo.
echo  Loading changes for !SHOW_SHA!...
echo  Note: Press 'Q' to exit the view.
echo.

call git show "!SHOW_SHA!"

echo.
pause
goto CatHistory

:DoSearchCommit
cls
echo.
echo ===========================================================
echo            S E A R C H   H I S T O R Y
echo ===========================================================
echo.
echo  Looking for a specific change? Enter a keyword from
echo  your save descriptions (e.g. 'login' or 'fix').
echo.

set "SEARCH_Q="
set /p "SEARCH_Q= Enter search keyword: "

if "!SEARCH_Q!"=="" (
    echo.
    echo Error: Search keyword cannot be empty.
    pause
    goto CatHistory
)

echo.
echo  Searching all branches for '!SEARCH_Q!'...
echo -----------------------------------------------------------
echo.

call git log --oneline --all --grep="!SEARCH_Q!"

echo.
echo -----------------------------------------------------------
echo  If you see the ID you want, copy it and use 'View Save Point'.
echo.
pause
goto CatHistory

:DoBlame
cls
echo.
echo ===========================================================
echo             L I N E   B Y   L I N E   H I S T O R Y
echo ===========================================================
echo.
echo  This shows who last changed every single line in a file.
echo.

set "BLAME_F="
set /p "BLAME_F= Enter the file path: "

if "!BLAME_F!"=="" (
    echo.
    echo Error: No file specified.
    pause
    goto CatHistory
)

echo.
echo  Loading history for !BLAME_F!...
echo  Note: Press 'Q' to exit the view.
echo.

call git blame "!BLAME_F!"

echo.
pause
goto CatHistory

:DoFileHistory
cls
echo.
echo ===========================================================
echo               F I L E   E V O L U T I O N
echo ===========================================================
echo.
echo  This shows a list of every save point that affected
echo  this specific file.
echo.

set "FHIST_F="
set /p "FHIST_F= Enter the file path: "

if "!FHIST_F!"=="" (
    echo.
    echo Error: No file specified.
    pause
    goto CatHistory
)

set "FHIST_N="
set /p "FHIST_N= How many save points to show? Enter for 10: "
if "!FHIST_N!"=="" set "FHIST_N=10"

echo.
echo  Showing the last !FHIST_N! changes to !FHIST_F!...
echo -----------------------------------------------------------
echo.

call git log -!FHIST_N! --oneline -- "!FHIST_F!"

echo.
echo -----------------------------------------------------------
echo.
pause
goto CatHistory

:DoReflog
cls
echo.
echo ===========================================================
echo             T I M E   M A C H I N E   (REFLOG)
echo ===========================================================
echo.
echo  This is a master record of every move you have made.
echo  Even if you deleted a branch or made a mistake, you can
echo  usually find the "lost" version here.
echo.

set "RLOG_N="
set /p "RLOG_N= How many moves to show? Enter for 20: "
if "!RLOG_N!"=="" set "RLOG_N=20"

echo.
echo  Note: Look for the {number} next to the ID to go back in time.
echo  Press 'Q' to exit the list.
echo -----------------------------------------------------------
echo.

call git reflog -!RLOG_N!

echo.
echo -----------------------------------------------------------
echo.
pause
goto CatHistory

:DoShortlog
cls
echo.
echo ===========================================================
echo             P R O J E C T   C O N T R I B U T O R S
echo ===========================================================
echo.
echo  This list shows everyone who has contributed to this
echo  project and how many save points they have created.
echo.
echo  Rank  ^|  Name and Email
echo -----------------------------------------------------------

call git shortlog -sne

echo -----------------------------------------------------------
echo.
pause
goto CatHistory

:DoShowFileAtCommit
cls
echo.
echo ===========================================================
echo          V I E W   F I L E   F R O M   P A S T
echo ===========================================================
echo.
echo  This lets you see what a specific file looked like at
echo  any point in your project's history.
echo.

set "SFC_SHA="
set /p "SFC_SHA= Enter the ID Number (Hash): "
set "SFC_FILE="
set /p "SFC_FILE= Enter the file path: "

if "!SFC_SHA!"=="" goto :SFC_Error
if "!SFC_FILE!"=="" goto :SFC_Error

echo.
echo  Loading !SFC_FILE! from save point !SFC_SHA!...
echo  Note: Press 'Q' to exit the view.
echo.

call git show "!SFC_SHA!":"!SFC_FILE!"

echo.
pause
goto CatHistory

:SFC_Error
echo.
echo  Error: Both an ID Number and a File Path are required.
pause
goto CatHistory

:DoCountCommits
cls
echo.
echo ===========================================================
echo             P R O J E C T   S T A T I S T I C S
echo ===========================================================
echo.
echo  Total save points created:
echo.

:: Count for the current branch
for /f %%C in ('git rev-list --count HEAD 2^>nul') do (
    echo  Current Branch:  %%C
)

:: Count for every branch combined
for /f %%C in ('git rev-list --count --all 2^>nul') do (
    echo  All Branches:    %%C
)

echo.
pause
goto CatHistory

:DoDiffCommits
cls
echo.
echo ===========================================================
echo           C O M P A R E   S A V E   P O I N T S
echo ===========================================================
echo.
echo  Recent History (Copy the IDs you need):
call git log -10 --oneline
echo.

set "DC_A="
set /p "DC_A= Enter the First ID: "
set "DC_B="
set /p "DC_B= Enter the Second ID: "

if "!DC_A!"=="" goto :DC_Error
if "!DC_B!"=="" goto :DC_Error

echo.
echo  Comparing changes between !DC_A! and !DC_B!...
echo  Note: Press 'Q' to exit the view.
echo.

call git diff "!DC_A!" "!DC_B!"

echo.
pause
goto CatHistory

:DC_Error
echo.
echo  Error: You must provide two IDs to compare.
pause
goto CatHistory

:DoLsFiles
cls
echo.
echo ===========================================================
echo            W H A T   G I T   I S   W A T C H I N G
echo ===========================================================
echo.
echo  This is a master list of every file currently being
echo  tracked and protected by your project's history.
echo.

call git ls-files

echo.
echo -----------------------------------------------------------
echo  Note: Files in your folder but not on this list are
echo  untracked (ignored) by Git.
echo.
pause
goto CatHistory

:DoCommitStats
cls
echo.
echo ===========================================================
echo             S A V E   P O I N T   I M P A C T
echo ===========================================================
echo.
echo  This shows exactly how many lines were added or removed
echo  in each of your recent save points.
echo.

set "CSTAT_N="
set /p "CSTAT_N= How many save points to analyze? Enter for 5: "
if "!CSTAT_N!"=="" set "CSTAT_N=5"

echo.
echo  Note: Press 'Q' to exit the view.
echo -----------------------------------------------------------
echo.

call git log -!CSTAT_N! --stat

echo.
echo -----------------------------------------------------------
echo.
pause
goto CatHistory

:CatMerge
cls
echo.
echo ===========================================================
echo                 C O M B I N E   B R A N C H E S
echo ===========================================================

set "STUCK_TYPE="
if exist ".git\MERGE_HEAD" set "STUCK_TYPE=MERGE"
if exist ".git\rebase-merge" set "STUCK_TYPE=REBASE"
if exist ".git\rebase-apply" set "STUCK_TYPE=REBASE"
if exist ".git\CHERRY_PICK_HEAD" set "STUCK_TYPE=CHERRY-PICK"

if defined STUCK_TYPE (
    echo.
    echo  ATTENTION: You are in the middle of a !STUCK_TYPE!
    echo  The following files have overlaps that need fixing:
    echo -----------------------------------------------------------
    :: This line finds files marked as "Unmerged" (the ones with conflicts)
    for /f "tokens=2" %%F in ('git status --porcelain ^| findstr "^UU ^AA ^DU ^UD"') do (
        echo    - %%F
    )
    echo -----------------------------------------------------------
    echo  INSTRUCTIONS:
    echo  1. Open these files and keep the code you want.
    echo  2. Use 'Resume' below once they are saved.
    echo  3. Or use 'Cancel' to quit and go back to normal.
    echo -----------------------------------------------------------
)

echo.
echo  [1]  Merge        - Grab work from another branch
echo  [2]  Rebase       - Re-stack your work on another branch
echo  [3]  Pick One     - Copy one specific save point (Cherry-pick)
echo  [4]  Squash       - Combine multiple save points into one
echo  [5]  Milestone    - Merge and always create a record (No-Fast-Forward)
echo.
echo  --- Fix or Cancel ---
echo  [6]  Resume       - Continue after fixing the files above
echo  [7]  Cancel       - Emergency Exit (Reset everything)
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number: "

if "!CH!"=="1" goto DoMerge
if "!CH!"=="2" goto DoRebase
if "!CH!"=="3" goto DoCherryPick
if "!CH!"=="4" goto DoSquash
if "!CH!"=="5" goto DoMergeNoFF
if "!CH!"=="6" goto DoResumeManager
if "!CH!"=="7" goto DoAbortManager
if "!CH!"=="0" goto MainMenu

goto CatMerge

:DoMerge
cls
echo.
echo ===========================================================
echo             C O M B I N E   B R A N C H E S
echo ===========================================================
echo.
for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
echo  You are currently standing on: !CURRENT_BRANCH!
echo.
echo  Available branches:
call git branch
echo.

echo  Step 1: Where is the work you want to grab?
set "SOURCE_BR="
set /p "SOURCE_BR= Enter the SOURCE branch name: "

if "!SOURCE_BR!"=="" (
    echo Error: You must specify a source branch.
    pause
    goto CatMerge
)

echo.
echo  Step 2: Where should this work go?

set "DEST_BR=main"
set /p "DEST_BR= Enter the DESTINATION branch (Press Enter for main): "

:: Ensure we aren't trying to merge a branch into itself
if "!DEST_BR!"=="!SOURCE_BR!" (
    echo.
    echo Error: Cannot merge !SOURCE_BR! into itself.
    pause
    goto CatMerge
)

call git add -A
echo.
:: Handle the automatic branch switch if necessary
if /I "!SOURCE_BR!"=="!CURRENT_BRANCH!" (
    call git diff --cached --quiet || call git commit -m "Final changes for !SOURCE_BR!"
)
if /I not "!SOURCE_BR!"=="!CURRENT_BRANCH!" if /I not "!DEST_BR!"=="!CURRENT_BRANCH!" (
    call git diff --cached --quiet || (
        set "BR_MSG="
        set /p "BR_MSG= Describe what you changed in !CURRENT_BRANCH!: "
        call git commit -m "!BR_MSG!"
    )

    echo  Switching you over to '!SOURCE_BR!'...
    call git checkout "!SOURCE_BR!"
    if errorlevel 1 (
        echo Error: Could not switch to !SOURCE_BR!.
        pause
        goto CatMerge
    )
)

if /I not "!DEST_BR!"=="!CURRENT_BRANCH!" (
    echo  Switching you over to '!DEST_BR!'...
    call git checkout "!DEST_BR!"
    if errorlevel 1 (
        echo Error: Could not switch to !DEST_BR!.
        pause
        goto CatMerge
    )
)

call git add -A

call git diff --cached --quiet || (
    set "BR_MSG="
    set /p "BR_MSG= Describe what you changed in !DEST_BR!: "
    call git commit -m "!BR_MSG!"
)

call git pull origin "!DEST_BR!" --no-rebase
if errorlevel 1 (
    echo.
    echo Pull failed - Overlaps found.
    call :ResolveConflicts
)

echo.
echo  Merging changes from '!SOURCE_BR!' into '!DEST_BR!'...
echo -----------------------------------------------------------
call git merge --no-ff "!SOURCE_BR!"

if errorlevel 1 (
    echo.
    echo  -----------------------------------------------------------
    echo  STUCK: Overlapping changes detected!
    echo  Launching the Conflict Resolution Tool...
    echo  -----------------------------------------------------------
    pause
    call :ResolveConflicts
) else (
    echo.
    echo  Success. !SOURCE_BR! has been combined into !DEST_BR!.
)

call git push origin "!DEST_BR!"

echo.
echo  Now that the work is combined, the task branch can be removed.
set "BR_CLEANUP="
set /p "BR_CLEANUP= Delete the branch !SOURCE_BR!? Y or N: "

if /I "!BR_CLEANUP!"=="Y" (

    echo Removing cloud !SOURCE_BR! branch...
    call git push origin --delete "!SOURCE_BR!" 2>nul

    echo Removing local !SOURCE_BR! branch...
    call git branch -d "!SOURCE_BR!"

    echo.
    echo Cleanup complete.
) else (
    echo !SOURCE_BR! kept for now.
)

echo.
echo Done. Your work is now part of !DEST_BR!.

echo.
pause
goto CatMerge

:DoRebase
cls
echo.
echo ===========================================================
echo            R E L O C A T E   W O R K   (REBASE)
echo ===========================================================
echo.
for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
echo  Target: Moving your current work on '!CURRENT_BRANCH!'
echo.
echo  Rebasing is a way to "clean up" your timeline so it looks
echo  like you started your work on the very latest version.
echo.

set "REBASE_BR="
set /p "REBASE_BR= Which branch should we build on top of? (e.g. main): "

if "!REBASE_BR!"=="" (
    echo Error: No branch name provided.
    pause
    goto CatMerge
)

echo.
echo  Is this work still PRIVATE to you, or have you PUBLISHED it?
echo.
echo  [PRIVATE]   I have NOT uploaded this work to the cloud yet.
echo              (It is safe to rewrite your own history)
echo.
echo  [PUBLIC] I have already UPLOADED this work to the cloud.
echo              (STOP: Use 'Merge' to avoid breaking sync for others)
echo.

set "REBASE_STATUS="
set /p "REBASE_STATUS= Type your status (PRIVATE or PUBLIC): "

if /I "!REBASE_STATUS!"=="PRIVATE" (
    echo.
    echo  Confirmed. Starting relocation onto !REBASE_BR!...
) else (
    echo.
    echo  Action Cancelled.
    echo  Since your work is already out there, 'Merge' is the
    echo  proper way to combine changes without breaking the project.
    pause
    goto CatMerge
)

call git rebase "!REBASE_BR!"

if errorlevel 1 (
    echo.
    echo  -----------------------------------------------------------
    echo  STUCK: Conflict detected during relocation!
    echo  -----------------------------------------------------------
    pause
    call :HandleRebaseConflicts
) else (
    echo.
    echo  Success. Your work has been relocated.
)

call :PromptForcePush

echo.
pause
goto CatMerge

:DoMergeNoFF
cls
echo.
echo ===========================================================
echo             M I L E S T O N E   M E R G E   (NO-FF)
echo ===========================================================
echo.
for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"
echo  You are currently standing on: !CURRENT_BRANCH!
echo.
echo  This type of merge creates a permanent "Merge Point" in your
echo  history, making it easy to see exactly when a feature was
echo  finished and added to the main project.
echo.

echo  Step 1: Where is the work you want to grab?
set "SOURCE_BR="
set /p "SOURCE_BR= Enter the SOURCE branch name: "

if "!SOURCE_BR!"=="" (
    echo Error: You must specify a source branch.
    pause
    goto CatMerge
)

echo.
echo  Step 2: Where should this work go?
set "DEST_BR="
set /p "DEST_BR= Enter the DESTINATION (Press Enter for !CURRENT_BRANCH!): "
if "!DEST_BR!"=="" set "DEST_BR=!CURRENT_BRANCH!"

:: Handle the automatic branch switch if necessary
if /I not "!DEST_BR!"=="!CURRENT_BRANCH!" (
    echo.
    echo  Switching you over to '!DEST_BR!' first...
    call git checkout "!DEST_BR!"
    if errorlevel 1 (
        echo.
        echo  Error: Could not switch branches. Save your work first!
        pause
        goto CatMerge
    )
)

echo.
echo  Creating a milestone merge from '!SOURCE_BR!' into '!DEST_BR!'...
echo -----------------------------------------------------------
call git merge --no-ff "!SOURCE_BR!"

if errorlevel 1 (
    echo.
    echo  -----------------------------------------------------------
    echo  STUCK: Overlapping changes detected!
    echo  Launching the Conflict Resolution Tool...
    echo  -----------------------------------------------------------
    pause
    call :ResolveConflicts
) else (
    echo.
    echo  Success. A new merge milestone was created in '!DEST_BR!'.
)

echo.
pause
goto CatMerge

:DoAbortManager
cls
echo.
echo ===========================================================
echo            S T U C K   P R O C E S S   M A N A G E R
echo ===========================================================
echo.
echo  Checking for active processes that can be cancelled...
echo.

set "MERGE_ACTIVE=0"
set "REBASE_ACTIVE=0"
set "COUNT=0"

:: Check for Merge
if exist ".git\MERGE_HEAD" (
    set "MERGE_ACTIVE=1"
    set /a COUNT+=1
    set "OPT_!COUNT!=MERGE"
    echo  [!COUNT!] Active MERGE detected - (You are currently combining branches)
)

:: Check for Rebase
if exist ".git\rebase-merge" set "REBASE_ACTIVE=1"
if exist ".git\rebase-apply" set "REBASE_ACTIVE=1"
if "!REBASE_ACTIVE!"=="1" (
    set /a COUNT+=1
    set "OPT_!COUNT!=REBASE"
    echo  [!COUNT!] Active REBASE detected - (You are currently relocating work)
)

if "!COUNT!"=="0" (
    echo  Everything looks normal. No stuck merges or rebases found.
    echo.
    pause
    goto CatMerge
)

echo.
echo  [X] Go back to menu (Do nothing)
echo.
set "ABORT_CHOICE="
set /p "ABORT_CHOICE= Select the number to cancel that process: "

if /I "!ABORT_CHOICE!"=="X" goto CatMerge

if defined OPT_!ABORT_CHOICE! (
    set "ACTION=!OPT_%ABORT_CHOICE%!"

    if "!ACTION!"=="MERGE" (
        echo.
        echo  Cancelling Merge...
        call git merge --abort
        echo  Done. Your files have been reset to before the merge started.
    )

    if "!ACTION!"=="REBASE" (
        echo.
        echo  Cancelling Rebase...
        call git rebase --abort
        echo  Done. Your files have been reset to before the rebase started.
    )
) else (
    echo Invalid selection.
)

echo.
pause
goto CatMerge

:DoResumeManager
cls
echo.
echo ===========================================================
echo             R E S U M I N G   P R O C E S S
echo ===========================================================
echo.

:: Detect the state again to choose the right command
set "RESUME_CMD="
if exist ".git\MERGE_HEAD" set "RESUME_CMD=merge --continue"
if exist ".git\rebase-merge" set "RESUME_CMD=rebase --continue"
if exist ".git\rebase-apply" set "RESUME_CMD=rebase --continue"
if exist ".git\CHERRY_PICK_HEAD" set "RESUME_CMD=cherry-pick --continue"

if not defined RESUME_CMD (
    echo  Nothing to resume. Everything looks like it is already finished.
    pause
    goto CatMerge
)

echo  Checking if all conflicts are marked as fixed...
:: Check if any files are still in an 'Unmerged' state
git status --porcelain | findstr "^UU ^AA ^DU ^UD" >nul
if not errorlevel 1 (
    echo.
    echo  STOP: You still have files with conflict markers.
    echo  Please fix them and 'Mark' them (git add) before resuming.
    pause
    goto CatMerge
)

echo.
echo  All clear. Finalizing the !RESUME_CMD!...
echo.

:: Execute the specific continue command
call git !RESUME_CMD! --no-edit

if errorlevel 1 (
    echo.
    echo  The resume failed. This sometimes happens if there are
    echo  more overlaps in the next set of changes.
    echo  Check the menu again for new conflicting files.
) else (
    echo.
    echo  Success. The process has been completed.
)

echo.
pause
goto CatMerge

:DoCherryPick
cls
echo.
echo ===========================================================
echo            G R A B   A   S A V E   P O I N T
echo ===========================================================
echo.
echo  Cherry-picking lets you copy one specific change from
echo  another branch without merging everything.
echo.
echo  Recent History (Across all branches):
echo -----------------------------------------------------------
call git log --all -10 --oneline
echo -----------------------------------------------------------
echo.

set "CP_SHA="
set /p "CP_SHA= Enter the ID Number (Hash) of the save to copy: "

if "!CP_SHA!"=="" (
    echo Error: No ID Number provided.
    pause
    goto CatMerge
)

echo.
echo  Attempting to copy save point !CP_SHA!...
call git cherry-pick "!CP_SHA!"

if errorlevel 1 (
    echo.
    echo  -----------------------------------------------------------
    echo  STUCK: Overlapping changes detected!
    echo  The copy is paused. Check the menu for conflicting files.
    echo  -----------------------------------------------------------
) else (
    echo.
    echo  Success. The change has been copied to your current branch.
)

echo.
pause
goto CatMerge

:DoSquash
cls
echo.
echo ===========================================================
echo             C L E A N   U P   H I S T O R Y
echo ===========================================================
echo.
echo  Squashing combines several small save points into one
echo  single, clean description. Great for hiding "Oops" saves.
echo.
echo  Recent History:
call git log -10 --oneline
echo.

set "SQ_N="
set /p "SQ_N= How many recent saves should be combined? (e.g. 3): "

if "!SQ_N!"=="" (
    echo Error: Number of saves is required.
    pause
    goto CatMerge
)

echo.
echo  QUICK CHECK:
echo  Is this work still PRIVATE to you, or have you PUBLISHED it?
echo.
echo  [PRIVATE]   I have NOT uploaded these saves to the cloud yet.
echo  [PUBLISHED] I have already UPLOADED these saves.
echo.

set "SQ_STATUS="
set /p "SQ_STATUS= Type your status (PRIVATE or PUBLISHED): "

if /I not "!SQ_STATUS!"=="PRIVATE" (
    echo.
    echo  Cancelled. You should not squash saves that are
    echo  already on the cloud, as it will break sync for others.
    pause
    goto CatMerge
)

echo.
echo  Combining the last !SQ_N! saves...
call git reset --soft HEAD~!SQ_N!

echo.
set "SQ_MSG="
set /p "SQ_MSG= Enter a new, clean description for this work: "
if "!SQ_MSG!"=="" set "SQ_MSG=Combined save point"

call git commit -m "!SQ_MSG!"

echo.
echo  Success. !SQ_N! saves have been condensed into one.
echo.

:: Since history was rewritten, a force-push is needed if they push later
call :PromptForcePush

echo.
pause
goto CatMerge

:DoCherryPickMulti
cls
echo.
echo ===========================================================
echo          C O P Y   A   R A N G E   O F   S A V E S
echo ===========================================================
echo.
echo  This lets you grab a whole sequence of changes at once.
echo.
echo  Recent History (Across all branches):
echo -----------------------------------------------------------
call git log --all -15 --oneline
echo -----------------------------------------------------------
echo.
echo  NOTE: To include the very first save point in your range,
echo  select the ID of the save IMMEDIATELY BEFORE it.
echo.

set "CPM_FROM="
set /p "CPM_FROM= Enter the ID BEFORE the start of your range: "
set "CPM_TO="
set /p "CPM_TO= Enter the ID of the LAST save in your range: "

if "!CPM_FROM!"=="" goto :CPM_Error
if "!CPM_TO!"=="" goto :CPM_Error

echo.
echo  Attempting to copy range from !CPM_FROM! to !CPM_TO!...
call git cherry-pick "!CPM_FROM!".."!CPM_TO!"

if errorlevel 1 (
    echo.
    echo  -----------------------------------------------------------
    echo  STUCK: Overlapping changes detected during the range!
    echo  The process is paused. Check the menu for conflicting files.
    echo  -----------------------------------------------------------
) else (
    echo.
    echo  Success. The range of changes has been copied.
)

echo.
pause
goto CatMerge

:CPM_Error
echo.
echo  Error: Both a Start and End ID are required.
pause
goto CatMerge

:DoCherryPickAbort
cls
echo.
echo ===========================================================
echo          C A N C E L   C O P Y   (ABORT)
echo ===========================================================
echo.
echo  Use this if the cherry-pick is too complex or you
echo  selected the wrong range.
echo.
echo  Your project will return to how it was before you
echo  tried to copy these changes.
echo.

call git cherry-pick --abort

if errorlevel 1 (
    echo  No active copy process was found to cancel.
) else (
    echo  Success. The copy process has been cancelled.
)

echo.
pause
goto CatMerge

:CatUndo
cls
echo.
echo ===========================================================
echo               H I S T O R Y   ^&   C L E A N U P
echo ===========================================================
echo.
echo  --- FIX ^& ADJUST ---                --- UNDO ^& ERASE ---
echo  [1] Change last description         [5] Undo save (Keep work)
echo  [2] Add forgotten files             [6] Erase save (Wipe work)
echo  [3] Cleanup untracked files         [7] Reverse a shared save
echo.
echo  --- TRAVEL ^& EXPLORE ---            --- RECOVERY ---
echo  [4] Travel to old version           [8] Resurrect deleted file
echo                                      [9] Emergency recovery
echo.
echo -----------------------------------------------------------
echo  [0] BACK TO MAIN MENU
echo -----------------------------------------------------------

set "CH="
set /p "CH= Select an option: "

if "!CH!"=="1" goto DoAmend
if "!CH!"=="2" goto DoAmendFiles
if "!CH!"=="3" goto DoClean
if "!CH!"=="4" goto DoResetTo
if "!CH!"=="5" goto DoSoftReset
if "!CH!"=="6" goto DoHardResetLast
if "!CH!"=="7" goto DoRevert
if "!CH!"=="8" goto DoRestoreDeleted
if "!CH!"=="9" goto DoReflogRecover
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice.
pause
goto CatUndo

:DoSoftReset
cls
echo.
echo ===========================================================
echo            U N D O   S A V E   (KEEP WORK)
echo ===========================================================
echo.
echo  This "un-saves" your work but keeps all your changes.
echo  Your files will stay exactly as they are right now.
echo.
echo  Recent History:
call git log -5 --oneline
echo.

set "SR_COMMIT="
set /p "SR_COMMIT= Enter the ID to undo (Press Enter for the last save): "
if "!SR_COMMIT!"=="" set "SR_COMMIT=HEAD"

echo.
echo  Un-saving !SR_COMMIT!...
call git reset --soft !SR_COMMIT!~1

echo.
echo  Success. The save point is gone, but your work is still
echo  here. You can find your files in the 'Waiting Area'.
echo.

:: Rewriting history often requires a force-push if previously uploaded
call :PromptForcePush

echo.
pause
goto CatUndo

:DoHardResetLast
cls
echo.
echo ===========================================================
echo            E R A S E   S A V E   (WIPE WORK)
echo ===========================================================
echo.
echo  Recent History:
call git log -5 --oneline
echo.

set "HR_COMMIT="
set /p "HR_COMMIT= Enter the ID to erase (Press Enter for the last save): "
if "!HR_COMMIT!"=="" set "HR_COMMIT=HEAD"

echo.
echo  ===== DANGER =====
echo  This will PERMANENTLY DELETE the work inside
echo  this save point and any unsaved changes.
echo  You cannot "Undo" this delete easily.
echo.

set "HR_CONFIRM="
set /p "HR_CONFIRM= Are you 100%% sure you want to WIPE this work? Y or N: "

if /I not "!HR_CONFIRM!"=="Y" (
    echo.
    echo  Cancelled. No files were harmed.
    pause
    goto CatUndo
)

echo.
echo  Wiping !HR_COMMIT! and resetting files...
call git reset --hard !HR_COMMIT!~1

echo.
echo  Done. The save point and its changes have been erased.
echo.

call :PromptForcePush

echo.
pause
goto CatUndo

:DoRevert
cls
echo.
echo ===========================================================
echo            R E V E R S E   A   S A V E   (SAFE)
echo ===========================================================
echo.
echo  This creates a NEW save point that does the exact opposite
echo  of an old one. It is the SAFEST way to undo changes that
echo  you have already shared with a team.
echo.
echo  Recent History:
call git log -5 --oneline
echo.

set "REV_SHA="
set /p "REV_SHA= Enter the ID Number (Hash) to reverse: "

if "!REV_SHA!"=="" (
    echo Error: No ID Number provided.
    pause
    goto CatUndo
)

echo.
echo  Creating a reversal for !REV_SHA!...
:: --no-edit prevents Git from forcing the user into a text editor
call git revert "!REV_SHA!" --no-edit

if errorlevel 1 (
    echo.
    echo  Note: If this failed, it is likely because newer changes
    echo  depend on the code you are trying to reverse.
) else (
    echo.
    echo  Success. A new "Undo" save point has been created.
)

echo.
pause
goto CatUndo

:DoResetTo
cls
echo.
echo ===========================================================
echo            T R A V E L   B A C K   I N   T I M E
echo ===========================================================
echo.
echo  This jumps your entire project back to a specific point
echo  in history.
echo.
echo  Recent History:
call git log -10 --oneline
echo.

set "RESET_SHA="
set /p "RESET_SHA= Enter the ID you want to jump back to: "

if "!RESET_SHA!"=="" (
    echo Error: Destination ID is required.
    pause
    goto CatUndo
)

echo.
echo  How should we handle the work you've done SINCE then?
echo.
echo   [1] KEEP IT   - Keep my work in the 'Waiting Area' (Soft)
echo   [2] UNSTAGE   - Keep my work, but 'Un-mark' it (Mixed)
echo   [3] ERASE IT  - Permanently delete all work since then (Hard)
echo.
set "RESET_MODE="
set /p "RESET_MODE= Select a number (1-3): "

if "!RESET_MODE!"=="1" (
    echo  Jumping back... (Work is safe)
    call git reset --soft "!RESET_SHA!"
)
if "!RESET_MODE!"=="2" (
    echo  Jumping back... (Work is safe)
    call git reset --mixed "!RESET_SHA!"
)
if "!RESET_MODE!"=="3" (
    echo.
    echo  WARNING: DESTRUCTIVE ACTION
    echo  This will erase every change made since !RESET_SHA!.
    set "RESET_CONFIRM="
    set /p "RESET_CONFIRM= Are you sure? Type Y to ERASE: "
    if /I "!RESET_CONFIRM!"=="Y" (
        call git reset --hard "!RESET_SHA!"
    ) else (
        echo Cancelled.
        pause
        goto CatUndo
    )
)

call :PromptForcePush
echo.
pause
goto CatUndo

:DoClean
cls
echo.
echo ===========================================================
echo             C L E A N U P   T E S T   F I L E S
echo ===========================================================
echo.
echo  This removes files that are NOT being tracked by Git
echo  (like temporary notes, test logs, or build files).
echo.
echo  Searching for untracked items...
echo -----------------------------------------------------------
call git clean -n -d
echo -----------------------------------------------------------
echo.

set "CLEAN_CONFIRM="
set /p "CLEAN_CONFIRM= Delete these files permanently? Y or N: "

if /I "!CLEAN_CONFIRM!"=="Y" (
    echo  Cleaning up...
    call git clean -f -d
    echo  Done. Untracked files removed.
) else (
    echo  Cancelled. No files were deleted.
)
echo.
pause
goto CatUndo

:DoAmend
cls
echo.
echo ===========================================================
echo            F I X   D E S C R I P T I O N
echo ===========================================================
echo.
echo  This lets you change the text of your very last save.
echo.
echo  Current Description:
call git log -1 --oneline
echo.

set "AMEND_MSG="
set /p "AMEND_MSG= Enter the NEW description: "

if "!AMEND_MSG!"=="" (
    echo Error: Description cannot be empty.
    pause
    goto CatUndo
)

echo.
echo  Updating description...
call git commit --amend -m "!AMEND_MSG!"

echo.
echo  Success. The last save point now has the new message.
echo.

:: Since this changes the 'ID Number' of the save, force-push is needed
call :PromptForcePush

echo.
pause
goto CatUndo

:DoAmendFiles
cls
echo.
echo ===========================================================
echo            A D D   M I S S I N G   F I L E S
echo ===========================================================
echo.
echo  Did you forget to include a file in your last save?
echo  This "sneaks" it in without creating a new save point.
echo.
echo  Current Last Save:
call git log -1 --oneline
echo.
echo  Available Files to Add:
call git status --short
echo.

set "AMF_FILE="
set /p "AMF_FILE= Which file should we add? (Type 'all' for everything): "

if "!AMF_FILE!"=="" (
    echo Error: No file specified.
    pause
    goto CatUndo
)

if /I "!AMF_FILE!"=="all" (
    echo  Marking all files...
    call git add -A
) else (
    echo  Marking !AMF_FILE!...
    call git add "!AMF_FILE!"
)

echo.
echo  Merging files into the last save...
call git commit --amend --no-edit

echo.
echo  Success. The last save point has been updated.
echo.

call :PromptForcePush

echo.
pause
goto CatUndo

:DoRestoreDeleted
cls
echo.
echo ===========================================================
echo            R E S U R R E C T   A   F I L E
echo ===========================================================
echo.
echo  This brings back a file you deleted from your folder.
echo.

set "RDEL_FILE="
set /p "RDEL_FILE= Enter the name/path of the deleted file: "

if "!RDEL_FILE!"=="" (
    echo Error: You must provide a file name.
    pause
    goto CatUndo
)

echo.
echo  Searching history for !RDEL_FILE!...
echo -----------------------------------------------------------
call git log --oneline --all -- "!RDEL_FILE!"
echo -----------------------------------------------------------
echo.

set "RDEL_SHA="
set /p "RDEL_SHA= Enter the ID to restore from (Press Enter for latest): "

if "!RDEL_SHA!"=="" (
    echo  Restoring latest version of !RDEL_FILE!...
    call git checkout HEAD -- "!RDEL_FILE!"
) else (
    echo  Restoring version !RDEL_SHA! of !RDEL_FILE!...
    call git checkout "!RDEL_SHA!" -- "!RDEL_FILE!"
)

echo.
echo  Success. Check your folder; the file should be back.
echo.
pause
goto CatUndo

:DoReflogRecover
cls
echo.
echo ===========================================================
echo            E M E R G E N C Y   R E C O V E R Y
echo ===========================================================
echo.
echo  This is your "Safety Net." Even if you deleted a branch
echo  or did a 'Hard Undo', Git keeps a secret log of where
echo  you have been for the last 30 days.
echo.
echo  Recent Movements (Look for the IDs on the left):
echo -----------------------------------------------------------
call git reflog -15
echo -----------------------------------------------------------
echo.

set "RREC_SHA="
set /p "RREC_SHA= Enter the ID Number (Hash) you want to rescue: "

if "!RREC_SHA!"=="" (
    echo Error: No ID Number provided.
    pause
    goto CatUndo
)

echo.
echo  How would you like to rescue this version?
echo.
echo   [1] NEW BRANCH - Create a safe new branch at this point (Best)
echo   [2] COPY HERE  - Pull this specific version into your current branch
echo.
set "RREC_MODE="
set /p "RREC_MODE= Select an option (1-2): "

if "!RREC_MODE!"=="1" (
    echo.
    set "RREC_BR="
    set /p "RREC_BR= Enter a name for the new 'Rescue' branch: "
    if "!RREC_BR!"=="" set "RREC_BR=rescued-work"

    echo  Creating branch and switching you over...
    call git checkout -b "!RREC_BR!" "!RREC_SHA!"
    echo.
    echo  Success. You are now standing on branch '!RREC_BR!'.
)

if "!RREC_MODE!"=="2" (
    echo.
    echo  Attempting to copy !RREC_SHA! to your current branch...
    call git cherry-pick "!RREC_SHA!"
    echo.
    echo  Done. If there were no overlaps, the work is now here.
)

echo.
pause
goto CatUndo

:CatTags
cls
echo.
echo ===========================================================
echo            V E R S I O N   M A N A G E M E N T
echo ===========================================================
echo.
echo  --- VIEW ^& INSPECT ---              --- CREATE NEW VERSION ---
echo  [1] View all versions               [3] Quick Label (Simple)
echo  [2] See version details             [4] Official Release (With notes)
echo                                      [5] Label a past save point
echo.
echo  --- CLOUD / UPLOAD ---              --- REMOVE / FIX ---
echo  [6] Upload one version              [8] Delete from this computer
echo  [7] Upload all versions             [9] Delete from the Cloud
echo.
echo -----------------------------------------------------------
echo  [0] BACK TO MAIN MENU
echo -----------------------------------------------------------

set "CH="
set /p "CH= Select an option: "

if "!CH!"=="1" goto DoTagList
if "!CH!"=="2" goto DoTagShow
if "!CH!"=="3" goto DoTagLight
if "!CH!"=="4" goto DoTagAnnotated
if "!CH!"=="5" goto DoTagCommit
if "!CH!"=="6" goto DoTagPush
if "!CH!"=="7" goto DoTagPushAll
if "!CH!"=="8" goto DoTagDelete
if "!CH!"=="9" goto DoTagDeleteRemote
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatTags

:DoTagList
cls
echo.
echo ===========================================================
echo             P R O J E C T   M I L E S T O N E S
echo ===========================================================
echo.
echo  These are the important version labels (like v1.0)
echo  you have placed on your project's history.
echo.
echo  Current Tags:
echo -----------------------------------------------------------
call git tag -n1
echo -----------------------------------------------------------
echo.
pause
goto CatTags

:DoTagLight
cls
echo.
echo ===========================================================
echo             Q U I C K   L A B E L   (LIGHT)
echo ===========================================================
echo.
echo  Use this for personal bookmarks or internal versions.
echo  It’s just a simple name for the current save point.
echo.

set "TLIGHT="
set /p "TLIGHT= Enter version name (e.g. v1.0.1-beta): "

if "!TLIGHT!"=="" (
    echo Error: Tag name cannot be empty.
    pause
    goto CatTags
)

echo.
echo  Applying label '!TLIGHT!'...
call git tag "!TLIGHT!"

echo.
echo  Success. Version '!TLIGHT!' has been marked.
echo.
pause
goto CatTags

:DoTagAnnotated
cls
echo.
echo ===========================================================
echo          O F F I C I A L   R E L E A S E   (HEAVY)
echo ===========================================================
echo.
echo  Use this for major milestones. It stores the date,
echo  your name, and a specific release message.
echo.

set "TANN="
set /p "TANN= Enter release version (e.g. v1.0.0): "
set "TANN_MSG="
set /p "TANN_MSG= Enter release notes (e.g. Initial public release): "

if "!TANN!"=="" (
    echo Error: Release version is required.
    pause
    goto CatTags
)

echo.
echo  Creating official release '!TANN!'...
call git tag -a "!TANN!" -m "!TANN_MSG!"

echo.
echo  Success. Official milestone '!TANN!' created.
echo.
pause
goto CatTags

:DoTagDelete
cls
echo.
echo ===========================================================
echo            R E M O V E   L O C A L   L A B E L
echo ===========================================================
echo.
echo  This deletes the version name from your computer.
echo  It does NOT delete the code inside that version.
echo.
echo  Current Local Versions:
call git tag
echo.

set "TDEL="
set /p "TDEL= Enter the version name to remove: "

if "!TDEL!"=="" (
    echo Error: No name provided.
    pause
    goto CatTags
)

echo.
echo  Removing '!TDEL!'...
call git tag -d "!TDEL!"

echo.
echo  Done. The label has been removed from this computer.
pause
goto CatTags

:DoTagPushOne
cls
echo.
echo ===========================================================
echo           U P L O A D   O N E   V E R S I O N
echo ===========================================================
echo.
echo  This sends one specific version label to the Cloud (GitHub).
echo.
echo  Current Versions:
call git tag
echo.

set "TPUSH="
set /p "TPUSH= Which version should we upload? "

if "!TPUSH!"=="" (
    echo Error: Name is required.
    pause
    goto CatTags
)

set "TPUSH_REM=origin"
set /p "TPUSH_REM= Remote Destination (Press Enter for 'origin'): "
if "!TPUSH_REM!"=="" set "TPUSH_REM=origin"

echo.
echo  Uploading version '!TPUSH!' to !TPUSH_REM!...
call git push "!TPUSH_REM!" "!TPUSH!"

echo.
pause
goto CatTags

:DoTagPushAll
cls
echo.
echo ===========================================================
echo           U P L O A D   A L L   V E R S I O N S
echo ===========================================================
echo.
echo  This will sync ALL your version labels with the Cloud.
echo.

set "TPALL_REM=origin"
set /p "TPALL_REM= Remote Destination (Press Enter for 'origin'): "
if "!TPALL_REM!"=="" set "TPALL_REM=origin"

echo.
echo  Syncing all versions to !TPALL_REM!...
call git push "!TPALL_REM!" --tags

echo.
echo  Success. All your milestones are now visible on the Cloud.
echo.
pause
goto CatTags

:DoTagDeleteRemote
cls
echo.
echo ===========================================================
echo            D E L E T E   F R O M   T H E   C L O U D
echo ===========================================================
echo.
echo  This removes a version label from the Cloud (GitHub).
echo  Use this if you uploaded a version by mistake or
echo  if you need to rename a release.
echo.

set "TDELR="
set /p "TDELR= Which version should we remove from the Cloud? "

if "!TDELR!"=="" (
    echo Error: Version name is required.
    pause
    goto CatTags
)

set "TDELR_REM=origin"
set /p "TDELR_REM= From which Remote? (Press Enter for 'origin'): "
if "!TDELR_REM!"=="" set "TDELR_REM=origin"

echo.
echo  WARNING: This will remove '!TDELR!' for everyone else too
set "TDELR_CONFIRM="
set /p "TDELR_CONFIRM= Are you sure you want to delete it? Y or N: "

if /I "!TDELR_CONFIRM!"=="Y" (
    echo  Deleting version '!TDELR!' from !TDELR_REM!...
    call git push "!TDELR_REM!" --delete "!TDELR!"
) else (
    echo  Cancelled. The version is still on the Cloud.
)
echo.
pause
goto CatTags

:DoTagShow
cls
echo.
echo ===========================================================
echo             V E R S I O N   D E T A I L S
echo ===========================================================
echo.
echo  Existing Versions:
call git tag
echo.

set "TSHOW="
set /p "TSHOW= Enter the version name to inspect: "

if "!TSHOW!"=="" (
    echo Error: No version name provided.
    pause
    goto CatTags
)

echo.
echo -----------------------------------------------------------
call git show "!TSHOW!" --summary
echo -----------------------------------------------------------
echo.
pause
goto CatTags

:DoTagPast
cls
echo.
echo ===========================================================
echo          L A B E L   A   P A S T   V E R S I O N
echo ===========================================================
echo.
echo  Forgot to tag a release? You can pick any save point
echo   from your history and give it a version name now.
echo.
echo  Recent History:
call git log -10 --oneline
echo.

set "TC_SHA="
set /p "TC_SHA= Enter the ID of the old save point: "
set "TC_TAG="
set /p "TC_TAG= Enter the new version name (e.g. v0.9): "

if "!TC_SHA!"=="" goto :TagPastError
if "!TC_TAG!"=="" goto :TagPastError

echo.
echo  Would you like to add a description/notes?
set "TC_ANN="
set /p "TC_ANN= (Y for Yes, N for a simple name only): "

if /I "!TC_ANN!"=="Y" (
    set "TC_MSG="
    set /p "TC_MSG= Enter the version notes: "
    call git tag -a "!TC_TAG!" "!TC_SHA!" -m "!TC_MSG!"
) else (
    echo  Applying simple label...
    call git tag "!TC_TAG!" "!TC_SHA!"
)

echo.
echo  Success. !TC_TAG! has been attached to save point !TC_SHA!.
echo.
pause
goto CatTags

:TagPastError
echo Error: Both the ID and the Version Name are required.
pause
goto CatTags

:CatSubmodules
cls
echo.
echo ===========================================================
echo                 S U B P R O J E C T S
echo ===========================================================
echo.
echo  A subproject is a separate Git project embedded inside
echo  your own. This is also known as a submodule.
echo.
echo  [1]  Add - Embed a new subproject inside this one
echo  [2]  Initialize - Set up existing subprojects
echo  [3]  Update - Pull the latest changes for subprojects
echo  [4]  Status - Check the version of each subproject
echo  [5]  Remove - Safely delete a subproject
echo  [6]  Repair - Fix subproject internet addresses
echo  [7]  Deep Download - Copy a project and all its subprojects
echo.
echo  [0]  Back to main menu
echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Enter a number 0-7: "

if "!CH!"=="1" goto DoSubAdd
if "!CH!"=="2" goto DoSubInit
if "!CH!"=="3" goto DoSubUpdate
if "!CH!"=="4" goto DoSubStatus
if "!CH!"=="5" goto DoSubDeinit
if "!CH!"=="6" goto DoSubSync
if "!CH!"=="7" goto DoSubClone
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatSubmodules

:DoSubAdd
cls
echo.
echo ===========================================================
echo            L I N K   E X T E R N A L   P R O J E C T
echo ===========================================================
echo.
echo  This "plugs in" another Git project as a folder inside
echo  your own. Perfect for shared libraries or tools.
echo.

set "SUBA_URL="
set /p "SUBA_URL= Enter the URL of the project to add: "
set "SUBA_PATH="
set /p "SUBA_PATH= Name the folder where it should live: "

if "!SUBA_URL!"=="" (
    echo Error: A repository URL is required.
    pause
    goto CatSubmodules
)

echo.
echo  Downloading and linking external project...
call git submodule add "!SUBA_URL!" "!SUBA_PATH!"

echo.
echo  Success. The external project is now linked at '!SUBA_PATH!'.
echo  Note: Don't forget to SAVE (Commit) this change!
echo.
pause
goto CatSubmodules

:DoSubInit
cls
echo.
echo ===========================================================
echo           W A K E   U P   S U B M O D U L E S
echo ===========================================================
echo.
echo  Just downloaded this project? Submodule folders often
echo  start out empty. This "wakes them up" and gets them
echo  ready to download their contents.
echo.

call git submodule init

echo.
echo  Ready. Now use 'Update' to actually download the files.
echo.
pause
goto CatSubmodules

:DoSubUpdate
cls
echo.
echo ===========================================================
echo            D O W N L O A D   U P D A T E S
echo ===========================================================
echo.
echo  This grabs the latest files for your linked projects.
echo.
echo  [1] Standard Update - Download missing files
echo  [2] Deep Update     - Update everything inside sub-folders
echo  [3] Latest Version  - Grab the absolute newest from the Cloud
echo.
set "SUBU_CHOICE="
set /p "SUBU_CHOICE= Select an option (1-3): "

if "!SUBU_CHOICE!"=="1" (
    echo  Updating...
    call git submodule update --init
)
if "!SUBU_CHOICE!"=="2" (
    echo  Deep updating (Recursive)...
    call git submodule update --init --recursive
)
if "!SUBU_CHOICE!"=="3" (
    echo  Grabbing latest versions from Cloud...
    call git submodule update --remote --merge
)

echo.
echo  Finished checking submodules.
echo.
pause
goto CatSubmodules

:DoSubStatus
cls
echo.
echo ===========================================================
echo            L I N K E D   P R O J E C T   S T A T U S
echo ===========================================================
echo.
echo  Check if your external projects are up-to-date.
echo  (-) = Not downloaded yet
echo  (+) = Newer version available elsewhere
echo  (U) = Conflict/Overlap detected
echo.
echo -----------------------------------------------------------
call git submodule status
echo -----------------------------------------------------------
echo.
pause
goto CatSubmodules

:DoSubDeinit
cls
echo.
echo ===========================================================
echo            U N L I N K   S U B M O D U L E
echo ===========================================================
echo.
echo  This "unplugs" the external project from your folder.
echo  The files will be removed, but the link remains in
echo  your project history so you can "Plug it in" later.
echo.
echo  Current Links:
call git submodule status
echo.

set "SUBD_PATH="
set /p "SUBD_PATH= Which folder path should we unplug? "

if "!SUBD_PATH!"=="" (
    echo Error: Folder path is required.
    pause
    goto CatSubmodules
)

set "SUBD_CONFIRM="
set /p "SUBD_CONFIRM= Remove these files from your computer? Y or N: "

if /I "!SUBD_CONFIRM!"=="Y" (
    echo  Unplugging...
    call git submodule deinit -f "!SUBD_PATH!"
    echo.
    echo  Done. The folder is now empty and the module is "asleep."
) else (
    echo  Cancelled.
)
echo.
pause
goto CatSubmodules

:DoSubSync
cls
echo.
echo ===========================================================
echo          R E P A I R   C O N N E C T I O N S
echo ===========================================================
echo.
echo  Use this if an external project changed its web address
echo  (URL). It updates your local settings to match the
echo  latest project configuration.
echo.

call git submodule sync --recursive

echo.
echo  Success. Connections have been refreshed.
echo.
pause
goto CatSubmodules

:DoSubClone
cls
echo.
echo ===========================================================
echo          D O W N L O A D   W I T H   L I N K S
echo ===========================================================
echo.
echo  This downloads a project AND all of its external
echo  linked sub-projects at the same time.
echo.

set "SUBC_URL="
set /p "SUBC_URL= Enter the Cloud URL to download: "

if "!SUBC_URL!"=="" (
    echo Error: URL is required.
    pause
    goto CatSubmodules
)

set "SUBC_DIR="
set /p "SUBC_DIR= Destination folder name (Press Enter for default): "

echo.
echo  Downloading project and all sub-projects...
if "!SUBC_DIR!"=="" (
    call git clone --recursive "!SUBC_URL!"
) else (
    call git clone --recursive "!SUBC_URL!" "!SUBC_DIR!"
)

echo.
echo  Success. Everything has been downloaded.
echo.
pause
goto CatSubmodules

:CatAdvanced
cls
echo.
echo ===========================================================
echo                A D V A N C E D   T O O L S
echo ===========================================================
echo.
echo  --- FIND ^& PORTABILITY ---          --- WORKSPACES (WORKTREE) ---
echo  [1] Search Code (Grep)              [6] Multi-Task (New Folder)
echo  [2] Find Bug (Bisect)               [7] List Active Folders
echo  [3] Export (Zip/Tar)                [8] Close Extra Folder
echo  [4] Create Patch File
echo  [5] Apply Patch File                --- MAINTENANCE ^& INFO ---
echo                                      [9] Check if file is Ignored
echo  --- SYSTEM ^& EXPERT ---             [10] Optimize (Clean Cache)
echo  [13] Custom Command                 [11] Health Check (Fsck)
echo  [14] View Exclusions                [12] Storage/Space Info
echo  [15] View Shortcuts (Aliases)
echo.
echo -----------------------------------------------------------
echo  [0] BACK TO MAIN MENU
echo -----------------------------------------------------------

set "CH="
set /p "CH= Enter a number 0-15: "

if "!CH!"=="1" goto DoGrep
if "!CH!"=="2" goto DoBisect
if "!CH!"=="3" goto DoArchive
if "!CH!"=="4" goto DoPatchCreate
if "!CH!"=="5" goto DoPatchApply
if "!CH!"=="6" goto DoWorktreeAdd
if "!CH!"=="7" goto DoWorktreeList
if "!CH!"=="8" goto DoWorktreeRemove
if "!CH!"=="9" goto DoCheckIgnore
if "!CH!"=="10" goto DoGC
if "!CH!"=="11" goto DoFsck
if "!CH!"=="12" goto DoCountObjects
if "!CH!"=="13" goto DoCustomCmd
if "!CH!"=="14" goto DoIgnoreRules
if "!CH!"=="15" goto DoListAliases
if "!CH!"=="0" goto MainMenu

echo.
echo Invalid choice. Press any key to try again...
pause >nul
goto CatAdvanced

:DoGrep
cls
echo.
echo ===========================================================
echo             S E A R C H   I N   C O D E
echo ===========================================================
echo.
echo  This quickly finds every line of code that contains
echo  your search term across your entire project.
echo.

set "GREP_Q="
set /p "GREP_Q= What word or phrase are you looking for? "

if "!GREP_Q!"=="" (
    echo Error: Search term is required.
    pause
    goto CatAdvanced
)

echo.
echo  Searching...
echo -----------------------------------------------------------
:: -n shows line numbers, -i makes it case-insensitive
call git grep -n -i "!GREP_Q!"
echo -----------------------------------------------------------
echo.
pause
goto CatAdvanced

:DoBisect
cls
echo.
echo ===========================================================
echo                B U G   H U N T E R   G P S
echo ===========================================================
echo.

:: --- AUTO-DETECT STATE ---
set "BISECT_ACTIVE=0"
if exist ".git\BISECT_START" set "BISECT_ACTIVE=1"

if "!BISECT_ACTIVE!"=="0" (
    echo  [ STATUS: READY TO START ]
    echo.
    echo  1. Identify a version that is BROKEN (usually right now).
    echo  2. Identify a version from the past that was WORKING.
    echo.
    echo  [1] START HUNT - Click here to begin
    echo  [0] CANCEL     - Go back
) else (
    echo  [ STATUS: HUNTING... ]
    echo  Git has moved your files to a specific point in time.
    echo  Test your code now: Is the bug there?
    echo.
    echo  [2] STILL BROKEN   - The bug is still here
    echo  [3] FIXED/WORKING  - The bug is gone in this version
    echo.
    echo  --- OPTIONS ---
    echo  [5] SKIP   - Can't test this one (e.g. it won't compile)
    echo  [4] STOP   - Give up and return to the present
)

echo.
echo -----------------------------------------------------------
set "CH="
set /p "CH= Action: "

:: --- EXECUTION LOGIC ---
if "!CH!"=="0" goto CatAdvanced

if "!CH!"=="1" (
    call git bisect start
    echo.
    echo  Step 1: Current version is usually the broken one.
    call git bisect bad
    echo  Step 2: Enter the ID of a version that worked:
    set "BS_GOOD="
    set /p "BS_GOOD= ID: "
    if not "!BS_GOOD!"=="" (
        call git bisect good !BS_GOOD!
    ) else (
        echo  Error: You must provide a working ID to start the search.
        call git bisect reset
    )
    pause
    goto DoBisect
)

if "!CH!"=="2" call git bisect bad & pause & goto DoBisect
if "!CH!"=="3" call git bisect good & pause & goto DoBisect
if "!CH!"=="5" call git bisect skip & pause & goto DoBisect
if "!CH!"=="4" call git bisect reset & pause & goto CatAdvanced

goto DoBisect

:DoArchive
cls
echo.
echo ===========================================================
echo             P A C K A G E   P R O J E C T
echo ===========================================================
echo.
echo  This creates a clean ZIP or TAR file of your project
echo  without including the bulky '.git' history folder.
echo  Perfect for sending your code to someone else.
echo.

set "ARC_NAME="
set /p "ARC_NAME= Name your file (e.g. project-v1.zip): "
if "!ARC_NAME!"=="" set "ARC_NAME=project-backup.zip"

set "ARC_FMT="
set /p "ARC_FMT= Choose format (zip or tar): "
if "!ARC_FMT!"=="" set "ARC_FMT=zip"

echo.
echo  Creating !ARC_FMT! archive named !ARC_NAME!...
call git archive --format="!ARC_FMT!" --output="!ARC_NAME!" HEAD

echo.
echo  Success. Archive created: !ARC_NAME!
echo.
pause
goto CatAdvanced

:DoPatchCreate
cls
echo.
echo ===========================================================
echo            E X P O R T   C H A N G E S   (PATCH)
echo ===========================================================
echo.
echo  This creates a portable file containing your changes.
echo  You can send this file to a teammate to "plug in"
echo  your work without using a Cloud branch.
echo.
echo  Recent History:
call git log -5 --oneline
echo.

set "PAT_N="
set /p "PAT_N= How many recent saves should be exported? (e.g. 1): "

if "!PAT_N!"=="" set "PAT_N=1"

echo.
echo  Creating patch files...
call git format-patch -!PAT_N!

echo.
echo  Success. Look for '.patch' files in your project folder.
echo.
pause
goto CatAdvanced

:DoPatchApply
cls
echo.
echo ===========================================================
echo            I M P O R T   C H A N G E S   (PATCH)
echo ===========================================================
echo.
echo  This takes a '.patch' file sent by someone else and
echo  applies their changes to your current files.
echo.

set "PAT_FILE="
set /p "PAT_FILE= Enter the name of the .patch file (e.g. 0001-fix.patch): "

if "!PAT_FILE!"=="" (
    echo Error: No file specified.
    pause
    goto CatAdvanced
)

echo.
echo  Checking if the patch fits your code...
call git apply --check "!PAT_FILE!"

if errorlevel 1 (
    echo.
    echo  STOP: This patch doesn't fit your current code version.
    echo  It might be meant for a different branch.
) else (
    echo  Patch looks good. Applying changes...
    call git apply "!PAT_FILE!"
    echo  Success. The changes have been added to your files.
)

echo.
pause
goto CatAdvanced

:DoWorktreeAdd
cls
echo.
echo ===========================================================
echo            O P E N   S E P A R A T E   F O L D E R
echo ===========================================================
echo.
echo  This creates a NEW folder on your computer where you can
echo  work on a DIFFERENT branch at the same time.
echo.

set "WT_PATH="
set /p "WT_PATH= Folder Name (e.g. ../fix-folder): "
set "WT_BR="
set /p "WT_BR= Branch to open there (e.g. main): "

if "!WT_PATH!"=="" goto :WT_Error
if "!WT_BR!"=="" goto :WT_Error

echo.
echo  Creating separate workspace...
call git worktree add "!WT_PATH!" "!WT_BR!"

echo.
echo  Success. You now have a second copy of your project
echo  at !WT_PATH! running the '!WT_BR!' branch.
echo.
pause
goto CatAdvanced

:DoWorktreeList
cls
echo.
echo ===========================================================
echo            A C T I V E   W O R K S P A C E S
echo ===========================================================
echo.
echo  This shows all the folders currently linked to this
echo  project's history.
echo.
echo -----------------------------------------------------------
call git worktree list
echo -----------------------------------------------------------
echo.
pause
goto CatAdvanced

:DoWorktreeRemove
cls
echo.
echo ===========================================================
echo            C L O S E   W O R K S P A C E
echo ===========================================================
echo.
echo  Finished with your separate folder? This unlinks it.
echo  Note: This will not delete the files if they have
echo  unsaved changes.
echo.
echo  Current Workspaces:
call git log --oneline -1 --all :: Quick refresh
call git worktree list
echo.

set "WT_DEL="
set /p "WT_DEL= Enter the path of the folder to remove: "

if "!WT_DEL!"=="" (
    echo Error: Path is required.
    pause
    goto CatAdvanced
)

echo.
echo  Unlinking workspace...
call git worktree remove "!WT_DEL!"

echo.
echo  Done. The connection has been removed.
pause
goto CatAdvanced

:WT_Error
echo.
echo  Error: You must provide both a folder path and a branch.
pause
goto CatAdvanced

:DoCheckIgnore
cls
echo.
echo ===========================================================
echo            W H Y   I S   I T   I G N O R E D ?
echo ===========================================================
echo.
echo  If a file isn't showing up in your saves, it might be
echo  blocked by your '.gitignore' file. Use this to find out
echo  exactly which rule is stopping it.
echo.

set "CI_FILE="
set /p "CI_FILE= Enter the file path to check: "

if "!CI_FILE!"=="" (
    echo Error: File path is required.
    pause
    goto CatAdvanced
)

echo.
echo  Checking rules for !CI_FILE!...
echo -----------------------------------------------------------
call git check-ignore -v "!CI_FILE!"

if errorlevel 1 (
    echo  RESULT: This file is NOT ignored by any rules.
) else (
    echo.
    echo  RESULT: Above is the specific rule blocking this file.
)
echo -----------------------------------------------------------
echo.
pause
goto CatAdvanced

:DoGC
cls
echo.
echo ===========================================================
echo            O P T I M I Z E   P R O J E C T
echo ===========================================================
echo.
echo  Is your project folder feeling slow or bulky?
echo  This "Garbage Collection" compresses your history
echo  and cleans up temporary files to save space.
echo.

call git gc --prune=now --aggressive

echo.
echo  Success. Your project database has been optimized.
echo.
pause
goto CatAdvanced

:DoFsck
cls
echo.
echo ===========================================================
echo             H E A L T H   C H E C K   (FSCK)
echo ===========================================================
echo.
echo  This verifies that your project data isn't corrupted.
echo  It checks every link in your history for integrity.
echo.

call git fsck --full

echo.
echo  Check complete. If you see 'dangling' items, don't worry—
echo  those are just ghost files Git hasn't cleaned up yet.
echo.
pause
goto CatAdvanced

:DoCountObjects
cls
echo.
echo ===========================================================
echo             S T O R A G E   R E P O R T
echo ===========================================================
echo.
echo  This shows exactly how much disk space your
echo  Git history is using.
echo.

call git count-objects -vH

echo.
pause
goto CatAdvanced

:DoCustomCmd
cls
echo.
echo ===========================================================
echo             C U S T O M   G I T   C O M M A N D
echo ===========================================================
echo.
echo  Expert Mode: Type any Git command manually.
echo  (Do NOT type 'git' at the start)
echo.

set "CUSTOM_CMD="
set /p "CUSTOM_CMD= git "

if "!CUSTOM_CMD!"=="" (
    echo.
    echo  Cancelled. No command entered.
    pause
    goto CatAdvanced
)

echo.
echo  --- EXECUTING: git !CUSTOM_CMD! ---
echo.
call git !CUSTOM_CMD!

echo.
echo  --- COMMAND FINISHED ---
echo.
pause
goto CatAdvanced

:DoIgnoreRules
cls
echo.
echo ===========================================================
echo            V I E W   I G N O R E   R U L E S
echo ===========================================================
echo.
echo  These are the files and folders Git is currently
echo  programmed to ignore.
echo.

if exist ".gitignore" (
    echo  --- Contents of .gitignore ---
    echo.
    type .gitignore
    echo.
    echo  ------------------------------
) else (
    echo  No '.gitignore' file found in this folder.
)

echo.
pause
goto CatAdvanced

:DoListAliases
cls
echo.
echo ===========================================================
echo             Y O U R   S H O R T C U T S
echo ===========================================================
echo.
echo  These are the custom 'Aliases' (shortcuts) you have
echo  configured in your Git settings.
echo.

:: We use a simple filter to make the output cleaner
call git config --get-regexp alias | findstr /V "alias." 2>nul
if errorlevel 1 (
    :: Fallback if findstr fails or no aliases exist
    call git config --get-regexp alias
)

echo.
pause
goto CatAdvanced

:ResolveConflicts
cls
echo.
echo ===========================================================
echo          F I X I N G   C O D E   C O N F L I C T S
echo ===========================================================
echo.
echo  Git found overlapping changes. You need to decide which
echo  version to keep for each file listed below.
echo.

:ConflictFileLoop
echo -----------------------------------------------------------
echo  Files waiting for a decision:
echo -----------------------------------------------------------
set "CONF_COUNT=0"
for /f "tokens=1,*" %%A in ('git status --porcelain 2^>nul') do (
    if "%%A"=="UU" (
        set /a CONF_COUNT+=1
        echo  [!CONF_COUNT!] %%B
        set "FILE_!CONF_COUNT!=%%B"
    )
    if "%%A"=="AA" (
        set /a CONF_COUNT+=1
        echo  [!CONF_COUNT!] %%B - Both of you added this file
        set "FILE_!CONF_COUNT!=%%B"
    )
    if "%%A"=="DU" (
        set /a CONF_COUNT+=1
        echo  [!CONF_COUNT!] %%B - You deleted it, they changed it
        set "FILE_!CONF_COUNT!=%%B"
    )
    if "%%A"=="UD" (
        set /a CONF_COUNT+=1
        echo  [!CONF_COUNT!] %%B - They deleted it, you changed it
        set "FILE_!CONF_COUNT!=%%B"
    )
)

echo.
if "!CONF_COUNT!"=="0" (
    echo  All conflicts are fixed!
    echo  Finalizing the merge...
    call git add -A
    if not "!RESOLVE_NO_COMMIT!"=="1" call git commit --no-edit
    echo.
    echo  Success. Everything is back in sync.
    pause
    goto :eof
)

echo  Options:
echo  - Enter the [Number] of the file to fix it
echo  - Type ABORT to cancel and go back to normal
echo.

set "FILE_NUM="
set /p "FILE_NUM= Select a number or action: "

if /I "!FILE_NUM!"=="ABORT" (
    echo  Cancelling merge...
    call git merge --abort
    pause
    goto :eof
)

:: Map the number back to the filename
if defined FILE_!FILE_NUM! (
    set "CONF_FILE=!FILE_!FILE_NUM!!"
) else (
    echo Invalid selection.
    goto ConflictFileLoop
)

cls
echo.
echo  Working on: !CONF_FILE!
echo -----------------------------------------------------------
echo  What should we do with this file?
echo.
echo  [1]  Keep MY version - Discard their changes
echo  [2]  Keep THEIR version - Discard my changes
echo  [3]  Look inside the file to decide
echo.
set "CONF_ACTION="
set /p "CONF_ACTION= Select: "

if "!CONF_ACTION!"=="1" (
    call git checkout --ours "!CONF_FILE!"
    call git add "!CONF_FILE!"
    echo Kept YOUR version.
)
if "!CONF_ACTION!"=="2" (
    call git checkout --theirs "!CONF_FILE!"
    call git add "!CONF_FILE!"
    echo Kept THEIR version.
)
if "!CONF_ACTION!"=="3" (
    echo.
    echo ===========================================================
    echo  COMPARING VERSIONS FOR: !CONF_FILE!
    echo ===========================================================
    echo.
    echo --- YOUR VERSION (Local) ---
    echo.
    call git show :2:"!CONF_FILE!"
    echo.
    echo -----------------------------------------------------------
    echo --- THEIR VERSION (Cloud) ---
    echo.
    call git show :3:"!CONF_FILE!"
    echo.
    echo -----------------------------------------------------------
    echo.
    echo  What do you want to do?
    echo  [1] Keep YOUR version
    echo  [2] Keep THEIR version
    echo  [3] I will edit the file manually to mix them
    echo.
    set "CONF_EDIT="
    set /p "CONF_EDIT= Select: "

    if "!CONF_EDIT!"=="1" (
        call git checkout --ours "!CONF_FILE!"
        call git add "!CONF_FILE!"
        echo Kept YOUR version.
    )
    if "!CONF_EDIT!"=="2" (
        call git checkout --theirs "!CONF_FILE!"
        call git add "!CONF_FILE!"
        echo Kept THEIR version.
    )
    if "!CONF_EDIT!"=="3" (
        echo.
        echo  Please open the file in your code editor.
        echo  You will see the markers there - use them as guides
        echo  to combine the code, then delete the marker lines.
        echo.
        echo  Once you save the file, come back here.
        pause
        call git add "!CONF_FILE!"
        echo File marked as fixed.
    )
)

goto ConflictFileLoop

:PromptForcePush
echo.
set "SYNC_REMOTE="
set /p "SYNC_REMOTE= Sync changes to the cloud? Y or N: "
if /I not "!SYNC_REMOTE!"=="Y" goto :eof

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

echo Safe-updating the cloud...
call git push origin "!CURRENT_BRANCH!" --force-with-lease

if not errorlevel 1 (
    echo Cloud is now in sync!
    goto :eof
)

echo.
echo WARNING: The cloud has changes that you do not have locally.
echo A standard force-push will overwrite those changes forever.
echo.
set "FORCE_RETRY="
set /p "FORCE_RETRY= Overwrite cloud changes anyway? Y or N: "

if /I "!FORCE_RETRY!"=="Y" (
    call git push origin "!CURRENT_BRANCH!" --force
    echo Cloud has been forcefully updated.
) else (
    echo Push cancelled.
)
goto :eof

:PromptForcePushHard
echo.
echo ###########################################################
echo  IMPORTANT: CLOUD OVERWRITE REQUESTED
echo ###########################################################
echo  If you are trying to remove sensitive data or fix a
echo  major mistake, a force-push is required to update
echo  the cloud version of this project.
echo.
echo  WARNING: This will replace the history on the server
echo  with your local version.
echo.
set "SYNC_REMOTE="
set /p "SYNC_REMOTE= Proceed with the cloud overwrite? Y or N: "

if /I not "!SYNC_REMOTE!"=="Y" (
    echo.
    echo Overwrite cancelled.
    goto :eof
)

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%I"

echo.
echo FORCE-updating the cloud...
call git push origin "!CURRENT_BRANCH!" --force
echo Cloud has been updated and history has been replaced.
goto :eof

:DropSpecificCommit
set "DROP_COMMIT=%~1"
for /f "delims=" %%H in ('git rev-parse HEAD 2^>nul') do set "HEAD_SHA=%%H"
for /f "delims=" %%T in ('git rev-parse "!DROP_COMMIT!" 2^>nul') do set "DROP_SHA=%%T"

if "!HEAD_SHA!"=="!DROP_SHA!" (
    echo This is the latest save point. Performing a hard reset...
    call git reset --hard HEAD~1
    echo Done. The save point and all its changes have been erased.
    goto :eof
)

echo Removing a save point from the middle of history...
echo All other save points that came after it will be preserved.
echo.
echo Attempting automatic removal...

call git rebase -X theirs --onto !DROP_SHA!~1 !DROP_SHA!

if not errorlevel 1 (
    echo Done. The save point has been removed from history.
    goto :eof
)

echo.
echo Automatic removal was not enough to fix the overlaps.
echo You will need to help resolve the differences manually.
echo.

call :HandleRebaseConflicts

echo.
echo Done. The save point has been removed from history.
goto :eof

:HandleRebaseConflicts
set "REBASE_ACTIVE=0"
for /f "delims=" %%X in ('git status 2^>nul ^| findstr /C:"rebase in progress"') do set "REBASE_ACTIVE=1"

if "!REBASE_ACTIVE!"=="0" goto :eof

echo.
echo Attempting to automatically fix remaining overlaps...
call git checkout --theirs . 2>nul
call git add -A 2>nul
call git rebase --continue 2>nul

set "REBASE_ACTIVE=0"
for /f "delims=" %%X in ('git status 2^>nul ^| findstr /C:"rebase in progress"') do set "REBASE_ACTIVE=1"

if "!REBASE_ACTIVE!"=="0" (
    echo Automatic fix succeeded!
    goto :eof
)

echo.
echo Manual help is required for this step.
set "RESOLVE_NO_COMMIT=1"
call :ResolveConflicts
set "RESOLVE_NO_COMMIT="

echo Continuing with the history update...
call git rebase --continue 2>nul

goto HandleRebaseConflicts

:ExitScript
cls
echo.
echo ===========================================================
echo.
echo             Thanks for using Git Manager!
echo.
echo ===========================================================
echo.
echo  Closing session...
echo.
timeout /t 1 /nobreak >nul
endlocal
exit /B 0
