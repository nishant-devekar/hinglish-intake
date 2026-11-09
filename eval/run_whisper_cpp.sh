#!/bin/zsh
# Baselines: what the pipeline does today, plus the same models with Silero VAD.
cd "${0:A:h}"
M=../models
for c in clips/*.wav; do
  n=${c:t:r}
  for cfg in "turbo-hi:$M/ggml-large-v3-turbo.bin:-l hi" \
             "turbo-auto:$M/ggml-large-v3-turbo.bin:-l auto" \
             "turbo-hi-vad:$M/ggml-large-v3-turbo.bin:-l hi --vad -vm $M/ggml-silero-v5.1.2.bin -mc 0" \
             "small-tr:$M/ggml-small.bin:-l hi -tr -mc 0 -bs 5"; do
    name=${cfg%%:*}; rest=${cfg#*:}; model=${rest%%:*}; args=${rest#*:}
    s=$(perl -MTime::HiRes=time -e 'print time')
    whisper-cli -m $model ${=args} -f $c -np -otxt -of results/$n.$name >/dev/null 2>&1
    e=$(perl -MTime::HiRes=time -e 'print time')
    printf '%s\t%s\t%.1f\n' $n $name $(( e - s )) >> results/timings.tsv
  done
done
