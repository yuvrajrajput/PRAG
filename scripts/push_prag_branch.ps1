# Push the PRAG branch to GitHub (run after creating the remote repo).
#
# 1. Open https://github.com/new
# 2. Repository name: PRAG
# 3. Owner: yuvrajrajput
# 4. Do NOT initialize with README (empty repo)
# 5. Run this script from the repo root:
#      powershell -File scripts\push_prag_branch.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git remote set-url origin "https://github.com/yuvrajrajput/PRAG.git"
git push -u origin PRAG
Write-Host "Done. Branch PRAG pushed to https://github.com/yuvrajrajput/PRAG"
