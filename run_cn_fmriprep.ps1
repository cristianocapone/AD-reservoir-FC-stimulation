#!/usr/bin/env pwsh
# Batched fmriprep for 224 CN subjects (5 batches of ~45)
# Work dir is cleared between batches to manage disk space.

$BIDS    = "C:/Users/user/Desktop/2026.AD_MotionCorrection/CN_bids"
$OUT     = "C:/Users/user/Desktop/2026.AD_MotionCorrection/fmriprep_output"
$WORK    = "C:/Users/user/Desktop/2026.AD_MotionCorrection/work_CN_bids"
$LIC     = "C:/Users/user/Desktop/2026.AD_MotionCorrection/license.txt"
$PATCH   = "C:/Users/user/Desktop/2026.AD_MotionCorrection/report_patched.py"
$NIREPORTS_TARGET = "/app/.pixi/envs/fmriprep/lib/python3.12/site-packages/nireports/assembler/report.py"

New-Item -ItemType Directory -Force -Path $WORK | Out-Null

$batches = @(
    # Batch 1 — subjects 1-45
    @("002S0295","002S0413","002S0685","002S0729","002S1155","002S1261","002S1268","002S1280",
      "002S2010","002S2043","002S2073","002S4171","002S4213","002S4219","002S4225","002S4229",
      "002S4237","002S4251","002S4262","002S4264","002S4270","002S4447","002S4473","002S4521",
      "002S4654","002S4746","002S4799","002S5018","002S5178","002S5230","002S5256","006S0498",
      "006S0731","006S4150","006S4153","006S4192","006S4346","006S4357","006S4363","006S4449",
      "006S4485","006S4515","006S4546","006S4679","006S4713"),

    # Batch 2 — subjects 46-90
    @("006S4867","006S4960","006S5153","010S4135","010S4345","010S4442","010S5163","012S4012",
      "012S4026","012S4094","012S4128","012S4188","012S4545","012S4643","012S4849","012S4987",
      "012S5121","012S5157","012S5195","012S5213","013S1186","013S2324","013S2389","013S4236",
      "013S4268","013S4395","013S4579","013S4580","013S4595","013S4616","013S4731","013S4768",
      "013S4791","013S4917","013S4985","013S5071","013S5137","013S5171","018S2133","018S2138",
      "018S2155","018S2180","018S4257","018S4313","018S4349"),

    # Batch 3 — subjects 91-135
    @("018S4399","018S4400","018S4597","018S4696","018S4733","018S4809","018S4868","018S4889",
      "018S5074","018S5240","018S5250","018S5262","019S4252","019S4285","019S4293","019S4367",
      "019S4477","019S4548","019S4549","019S4680","019S4835","019S5012","019S5019","019S5180",
      "019S5242","031S0618","031S2017","031S2018","031S2022","031S2233","031S4005","031S4021",
      "031S4024","031S4029","031S4032","031S4042","031S4149","031S4194","031S4203","031S4218",
      "031S4474","031S4476","031S4496","031S4590","031S4721"),

    # Batch 4 — subjects 136-180
    @("031S4947","041S5026","053S0919","053S2357","053S2396","053S4557","053S4578","053S4661",
      "053S4813","053S5070","053S5202","053S5208","053S5272","053S5287","053S5296","100S0069",
      "100S0296","100S1286","100S2351","100S4469","100S4511","100S4512","100S4556","100S4884",
      "100S4970","100S5075","100S5091","100S5096","100S5102","100S5106","100S5246","100S5280",
      "129S0778","129S4073","129S4220","129S4287","129S4369","129S4371","129S4396","129S4422",
      "130S0969","130S2373","130S2391","130S2402","130S2403"),

    # Batch 5 — subjects 181-224
    @("130S4250","130S4294","130S4343","130S4352","130S4405","130S4415","130S4417","130S4468",
      "130S4542","130S4589","130S4605","130S4641","130S4660","130S4730","130S4817","130S4883",
      "130S4925","130S4971","130S4982","130S4984","130S4990","130S4997","130S5006","130S5059",
      "130S5142","130S5175","130S5231","130S5258","131S5138","131S5148","136S0107","136S0186",
      "136S4189","136S4269","136S4408","136S4433","136S4517","136S4726","136S4727","136S4836",
      "136S4848","136S4932","136S4956","136S4993")
)

$totalBatches = $batches.Count

for ($b = 0; $b -lt $totalBatches; $b++) {
    $batchNum = $b + 1
    $labels   = $batches[$b]
    $labelStr = $labels -join " "
    Write-Host ""
    Write-Host "========================================"
    Write-Host "BATCH $batchNum / $totalBatches  ($($labels.Count) subjects)"
    Write-Host "========================================"

    $argList = ($labels | ForEach-Object { "--participant-label $_" }) -join " "

    docker run --rm `
      -v "${BIDS}:/data:ro" `
      -v "${OUT}:/out" `
      -v "${WORK}:/tmp/work" `
      -v "${LIC}:/license.txt:ro" `
      nipreps/fmriprep:latest `
      /data /out participant `
      --participant-label @labels `
      --output-spaces MNI152NLin2009cAsym:res-native `
      --fs-license-file /license.txt `
      --fs-no-reconall `
      --work-dir /tmp/work `
      --nprocs 32 `
      --omp-nthreads 8 `
      --mem-mb 38000 `
      --skip-bids-validation

    $exitCode = $LASTEXITCODE
    Write-Host "Batch $batchNum exit code: $exitCode"

    # Clean work dir between batches (keeps directory, removes contents)
    Write-Host "Cleaning work directory..."
    Get-ChildItem $WORK -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Work dir cleaned."
}

Write-Host ""
Write-Host "========================================"
Write-Host "All batches done. Generating reports..."
Write-Host "========================================"

docker run --rm `
  -v "${BIDS}:/data:ro" `
  -v "${OUT}:/out" `
  -v "${WORK}:/tmp/work" `
  -v "${LIC}:/license.txt:ro" `
  -v "${PATCH}:${NIREPORTS_TARGET}:ro" `
  nipreps/fmriprep:latest `
  /data /out participant `
  --output-spaces MNI152NLin2009cAsym:res-native `
  --fs-license-file /license.txt `
  --fs-no-reconall `
  --work-dir /tmp/work `
  --reports-only `
  --skip-bids-validation

Write-Host "Done. Exit code: $LASTEXITCODE"
