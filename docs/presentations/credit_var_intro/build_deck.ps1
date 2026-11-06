<#
.SYNOPSIS
    Builds a dated PDF from the Credit VaR Marp deck.

.DESCRIPTION
    Copies the deck to _build.md, replaces {{DATE}} with today's date, and runs
    Marp CLI on the copy. The source deck is never changed. The script writes
    only _build.md and the PDF, both in the script folder.

    Marp CLI is found in this order:
      1. a marp command on PATH
      2. npx (part of Node.js), which runs @marp-team/marp-cli
    If neither exists, use -PrepareOnly. The script then writes _build.md and
    stops. Open _build.md in VS Code and export the PDF with the Marp extension.

    PDF export starts a browser. If the browser fails to start, try
    -Browser edge, -Browser chrome or -Browser firefox, or give the program
    file with -BrowserPath. If every browser fails, use -Html. The script then
    writes a standalone HTML file. Open it in a browser and print it to PDF.

.EXAMPLE
    .\build_deck.ps1
    Uses the highest-numbered CREDIT_VAR_INTRO_V*.md in this folder.

.EXAMPLE
    .\build_deck.ps1 -Source CREDIT_VAR_INTRO_V09.md -Output Credit_VaR_intro.pdf

.EXAMPLE
    .\build_deck.ps1 -PrepareOnly

.EXAMPLE
    .\build_deck.ps1 -Browser edge

.EXAMPLE
    .\build_deck.ps1 -BrowserPath "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

.EXAMPLE
    .\build_deck.ps1 -Html
#>
[CmdletBinding()]
param(
    [string]$Source,
    [string]$Output = "CREDIT_VAR_INTRO.pdf",
    [string]$DateFormat = "yyyy-MM-dd",
    [ValidateSet("auto", "chrome", "edge", "firefox")]
    [string]$Browser = "auto",
    [string]$BrowserPath,
    [switch]$Html,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$folder = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($folder)) {
    throw "Run this file as a script (.\build_deck.ps1). Do not paste it into the console."
}

# 1. Find the source deck.
if ([string]::IsNullOrWhiteSpace($Source)) {
    $candidate = Get-ChildItem -Path $folder -Filter "CREDIT_VAR_INTRO_V*.md" -File |
    Where-Object { $_.Name -match '_V(\d+)\.md$' } |
    Sort-Object { [int]([regex]::Match($_.Name, '_V(\d+)\.md$').Groups[1].Value) } -Descending |
    Select-Object -First 1
    if (-not $candidate) {
        throw "No CREDIT_VAR_INTRO_V<number>.md file found in $folder. Use -Source."
    }
    $srcPath = $candidate.FullName
}
else {
    if ([System.IO.Path]::IsPathRooted($Source)) { $srcPath = $Source } else { $srcPath = Join-Path $folder $Source }
    if (-not (Test-Path -LiteralPath $srcPath)) { throw "Source file not found: $srcPath" }
}

if ([System.IO.Path]::IsPathRooted($Output)) { $pdfPath = $Output } else { $pdfPath = Join-Path $folder $Output }
$buildPath = Join-Path $folder "_build.md"

# 2. Write the dated copy (UTF-8 without byte order mark, so the front matter still parses).
$date = (Get-Date).ToString($DateFormat, [System.Globalization.CultureInfo]::GetCultureInfo("en-GB"))
$text = [System.IO.File]::ReadAllText($srcPath, [System.Text.Encoding]::UTF8)
if (-not $text.Contains("{{DATE}}")) {
    Write-Warning "No {{DATE}} placeholder in $srcPath. The PDF will show no date."
}
$text = $text.Replace("{{DATE}}", $date)
[System.IO.File]::WriteAllText($buildPath, $text, (New-Object System.Text.UTF8Encoding $false))

Write-Host "Source: $srcPath"
Write-Host "Date:   $date"

if ($PrepareOnly) {
    Write-Host "Wrote:  $buildPath"
    Write-Host "Next:   open _build.md in VS Code and export the PDF with the Marp extension."
    return
}

# 3. Find Marp CLI. -CommandType Application skips the npm .ps1 shims, which the
#    script execution policy can block. It selects the .cmd or .exe file instead.
$marp = Get-Command marp -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$npx = Get-Command npx  -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $marp -and -not $npx) {
    throw ("Neither 'marp' nor 'npx' was found on PATH. Install Node.js (https://nodejs.org), " +
        "or run this script with -PrepareOnly and export the PDF from VS Code.")
}

# 4. Build the output.
#    PDF: --allow-local-files lets Marp load the images in the images folder.
#    HTML: no browser starts. The images load from the images folder next to the file.
if ($Html) {
    $outPath = [System.IO.Path]::ChangeExtension($pdfPath, ".html")
    $marpArgs = @($buildPath, "--html", "-o", $outPath)
}
else {
    $outPath = $pdfPath
    $marpArgs = @($buildPath, "--html", "--allow-local-files", "--pdf", "-o", $outPath)
    if ($Browser -ne "auto") { $marpArgs += @("--browser", $Browser) }
    if (-not [string]::IsNullOrWhiteSpace($BrowserPath)) { $marpArgs += @("--browser-path", $BrowserPath) }
}
if ($marp) {
    Write-Host "Using:  $($marp.Source)"
    & $marp.Source @marpArgs
}
else {
    Write-Host "Using:  npx @marp-team/marp-cli (npm may ask to install it)"
    & $npx.Source (@('@marp-team/marp-cli') + $marpArgs)
}
if ($LASTEXITCODE -ne 0) { throw "Marp failed with exit code $LASTEXITCODE." }

Write-Host "Built:  $outPath"
if ($Html) {
    Write-Host "Next:   open the HTML file in a browser, press Ctrl+P, choose Save as PDF, and turn on Background graphics."
}
