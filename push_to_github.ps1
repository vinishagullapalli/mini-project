# PowerShell script to push project to GitHub
$env:PATH = "$env:PATH;C:\Users\vinisha\.mingit\cmd"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Pushing Mini Project to GitHub Repository" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$token = Read-Host "Please enter your GitHub Personal Access Token (PAT)"

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "No token entered. Attempting standard git push..." -ForegroundColor Yellow
    git push -u origin main
} else {
    Write-Host "Pushing using provided token..." -ForegroundColor Green
    $remoteUrlWithToken = "https://$token@github.com/vinishagullapalli/mini-project.git"
    git push $remoteUrlWithToken main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSUCCESS! Code pushed to GitHub successfully!" -ForegroundColor Green
    } else {
        Write-Host "`nPush failed. Please verify that your token has 'repo' permissions." -ForegroundColor Red
    }
}
