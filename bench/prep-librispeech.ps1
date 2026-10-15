# Downloads LibriSpeech test-clean, converts FLAC to 16 kHz mono WAV, writes bench/data/test-clean/manifest.jsonl.
# Requires ffmpeg on PATH.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$data = Join-Path $PSScriptRoot 'data'
New-Item -ItemType Directory -Force $data | Out-Null
$tgz = Join-Path $data 'test-clean.tar.gz'
if (-not (Test-Path $tgz)) {
  Invoke-WebRequest 'https://www.openslr.org/resources/12/test-clean.tar.gz' -OutFile $tgz
}
$libri = Join-Path $data 'LibriSpeech'
if (-not (Test-Path $libri)) {
  tar -xzf $tgz -C $data
  if ($LASTEXITCODE -ne 0) {
    # Do not leave a partial extraction behind; the next run would skip extraction and miss files.
    Remove-Item -Recurse -Force $libri -ErrorAction SilentlyContinue
    throw "tar failed with exit code $LASTEXITCODE"
  }
}

$out = Join-Path $data 'test-clean'
New-Item -ItemType Directory -Force $out | Out-Null
$lines = [System.Collections.Generic.List[string]]::new()
Get-ChildItem (Join-Path $data 'LibriSpeech/test-clean') -Recurse -Filter *.trans.txt | ForEach-Object {
  $dir = $_.DirectoryName
  Get-Content $_.FullName | ForEach-Object {
    $id, $text = $_ -split ' ', 2
    $wav = Join-Path $out "$id.wav"
    if (-not (Test-Path $wav)) {
      # Write to a temp name and move into place, so an interrupted conversion never leaves a file that looks complete.
      $tmp = "$wav.tmp.wav"
      ffmpeg -loglevel error -y -i (Join-Path $dir "$id.flac") -ar 16000 -ac 1 -c:a pcm_s16le $tmp
      if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on $id with exit code $LASTEXITCODE" }
      Move-Item -Force $tmp $wav
    }
    $lines.Add((@{ audio = "$id.wav"; text = $text } | ConvertTo-Json -Compress))
  }
}
Set-Content (Join-Path $out 'manifest.jsonl') $lines
"wrote $($lines.Count) items"
