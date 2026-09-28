
# Frozen image-domain cohort

The manifest was frozen before route inspection or generated-output inspection. It contains **200** unique image hashes and uses balanced domain counts.

| domain | frozen images | executed EP4 | median visual tokens | median output tokens | EOS before cap |
|---|---:|---:|---:|---:|---:|
| docci | 40 | 40 | 3072.0 | 256.0 | 0 |
| textcaps | 40 | 40 | 672.0 | 256.0 | 1 |
| nocaps | 40 | 40 | 752.0 | 256.0 | 2 |
| dense_relation | 40 | 40 | 196.0 | 256.0 | 6 |
| scene_layout | 40 | 40 | 176.0 | 256.0 | 3 |

Source deviations are explicit: `{"dense_relation": "MMStar structured-diagram/layout public images substitute for unavailable paragraph-caption image assets", "scene_layout": "MMStar general visual reasoning public images substitute for unavailable ADE20K Localized Narratives pixels"}`.
The dense-relation and scene/layout lanes are MMStar public-image substitutes, not claims of exact Image-Paragraph-Captioning or ADE20K reproduction.

Generation sanity: malformed=0, refusals=0, prompt leakage=0, mean repeated-4gram fraction=0.0196. CIDEr/BLEU/SPICE are not presented as official scores because this universal prompt is not each dataset's official caption protocol.
