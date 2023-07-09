# PhoneLM

## About

Text to speech using phonemes as inputs and audio codec codes as outputs. Loosely based on MegaByte and VALL-E.

## Background

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

This is inspired by the fact that the authors of the original [MegaByte](https://arxiv.org/pdf/2305.07185.pdf) paper perform autoregressive audio prediction on raw audio data. They
treat the audio files as just raw byte sequences and train a model to predict audio on 2TB
worth of audio and find that compared to a vanilla transformer or Perceiver architectures,
it scores a higher bpb. In principle, this means that the model is more efficient and accurate
at modelling raw audio byte sequences than other approaches. The other benefits of the method
is that the patch based auto-regressive generation may be well suited to the codebooks used
by [encodec](https://github.com/facebookresearch/encodec). As the patch size can be set to 4
(for 4 codebooks each of which can be 1 of 1024 values), this means the local model of the
MegaByte model can focus on modelling individual audio codec elements and the global model
can focus on the larger context. Hopefully this greatly improves audio quality compared to
VALL-E.

## Method

- [ ] Use [encodec](https://github.com/facebookresearch/encodec) to
  encode and decode audio.

<!--
## Datasets

### LJSpeech
-->