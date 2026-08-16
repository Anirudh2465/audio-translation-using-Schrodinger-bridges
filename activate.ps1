# activate.ps1 -- Manual one-shot activator (double-click or run in PS)
# Usage: . .\activate.ps1   (note the leading dot for current shell)
$root = Split-Path $MyInvocation.MyCommand.Path
& "$root\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  EOT Speech Morphing venv activated." -ForegroundColor Cyan
Write-Host "  Python: $(python --version)" -ForegroundColor Green
Write-Host ""
