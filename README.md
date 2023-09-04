# PhoneLM

## About

UPDATE UPDATE (04/09/2023): Model overfitting on single and multiple samples works.
Generalisation seems to be harder. Probably because original MegaByte model from
lucidrains trained to predict entire sequence from 1 token to n-1 tokens. Need to
training method to only predict the response and not the token, otherwise the model
is having to pointlessly learn to predict the prompt as well.

UPDATE: Generalisation training seems some what promising. Model consistently outputs
the correct number of audio tokens and can deal with the temporal context somewhat well.
However, main issue seems to be more with the "spatial" component of predicting the sequence,
i.e., predicting the correct codebook codes per timestep.

Text to speech using phonemes as inputs and audio codec codes as outputs. Loosely based on MegaByte, VALL-E and Encodec.

## Method

- [x] Use [G2P](https://github.com/Kyubyong/g2p/) to encode text.
- [x] Use [encodec](https://github.com/facebookresearch/encodec) to
  encode and decode audio.
- [x] Custom LJSpeech dataloader to include phonemes and encodec audio codes

### LJSpeech

- [x] Overfit model on one sample from LJSpeech
  - [x] Combine token space of text and audio codec codes
  - `LJ016-0073-synth.wav` The initial "Mr. Cope" can just about be made out
  - Using a codebook of 2 seems to be too aggressive.
  - `LJ003-0259-synth.wav` "And attracted attention by their". Codebook of 2 is possible.
    Main issues is sequence length.
  - Scaling up sequence length is easier than scaling up codebook size. This is for the
    arrangement of [time1_code_1, time_1_code_2, ...].
    Perhaps [time1_code_1, time_2_code_1, ...] might perform better? So synthesize all codebook1 then all codebook 2.
  - Longer duration prompts and audio targets seem to perform worse. Will try experimenting
    with shorter prompts (try to stick to roughly 3 second audio snippets.)
- [-] Generalise (Using either 1 second prompt + clip, or 1.5 sec prompt and clip)
   - [x] Get any prompt to audio working (even if unintelligible and using clamping)
   - [-] Get any coherent output
   
<!--
## Datasets

### LJSpeech
-->

## Inspiration

This model is loosely based on the VALL-E paper by Microsoft. It uses the
MegaByte inspired model from [Lucidrains](https://github.com/lucidrains/MEGABYTE-pytorch)
as the Transformer Decoder model. Just as in VALL-E, a users text prompt is converted
into phonemes using [G2P](https://github.com/Kyubyong/g2p/) (Grapheme-to-phoneme),
and then the [encodec](https://github.com/facebookresearch/encodec) audio codec codes
are predicted. However, unlike VALL-E, only an autoregressive model is used. The VALL-E
paper uses an autoregressive model to accept phonemes and audio codec code snippets of
a source audio and uses that to predict the first codebook codes. The rest of the codebook
codes are then predicted when the AR model is finished, it accepts the entire sequence,
and then predicts all of the codebook 2 to codebook N codes. However, this increases
the complexity of the approach as two models are now required and raises the possibility
that the NAR model can not attend to all past inputs unlike the AR which can reduce
audio quality output and may lead to repeating of outputs. In practice, the use of phonemes
as input into VALL-E may alleviate this, however, this approach explores just predicting
the entire sequence auto-regressively (across all codebooks at once).

This is inspired by the fact that the authors of the original [MegaByte](https://arxiv.org/pdf/2305.07185.pdf)
paper perform autoregressive audio prediction on raw audio data. They
treat the audio files as just raw byte sequences and train a model to predict audio on 2TB
worth of audio and find that compared to a vanilla transformer or Perceiver architectures,
it scores a higher bpb. In principle, this means that the model is more efficient and accurate
at modelling raw audio byte sequences than other approaches. The other benefits of the method
is that the patch based auto-regressive generation may be well suited to the codebooks used
by [encodec](https://github.com/facebookresearch/encodec). As the patch size can be set to 4
(for 4 codebooks each of which can be 1 of 1024 values), this means the local model of the
MegaByte model can focus on modelling individual audio codec elements and the global model
can focus on the larger context. Hopefully this greatly improves audio quality compared to
VALL-E while being much simpler to train.