---
layout: post
title: Evolution of generative visual models
date: 2026-08-20
description: From Helmholtz machines and VAEs to GANs, diffusion, and visual foundation models — a historically cited walk through how we learned to sample images.
tags: generative-models diffusion foundation-models
categories: research
featured: true
related_posts: true
toc:
  beginning: true
---

**Disclaimer.** This text was generated with the help of an AI language model. I am the author: I chose the scope, the papers, and the argument, and I am responsible for what it claims. If a citation looks too neat, open the PDF.

---

Generative visual models are functions that sample from a distribution over images — or, more recently, over images, video, and other visual signals jointly with language. That sentence is easy to write in 2026. It was not obvious in 2013, when a "generated image" was a blurry MNIST digit, and it was not obvious in 1995, when the Helmholtz machine treated perception as inverting a generative model of sensory data ([Dayan, Hinton, Neal & Zemel, 1995](https://www.cs.toronto.edu/~hinton/absps/helmholtz.pdf)).

This is a walk through that path: density models, adversarial games, autoregressive pixels, diffusion, and the foundation-model era. I am not trying to be encyclopedic. I am trying to make the _lineage_ visible — why each generation looked like a dead end until the next one reused its parts.

## What "generative" used to mean

Before neural nets took over, vision already had generative stories. Sparse coding argued that natural images are produced by a small number of causes, and that V1 simple cells look the way they do because they invert that process ([Olshausen & Field, 1996](https://www.nature.com/articles/381607a0)). Boltzmann machines and restricted Boltzmann machines treated an image as a visible layer coupled to latent factors ([Smolensky, 1986](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf); [Hinton, 2002](https://www.cs.toronto.edu/~hinton/absps/nccd.pdf)). Deep belief networks stacked those layers and, briefly, made unsupervised pretraining the default way to train a deep net ([Hinton, Osindero & Teh, 2006](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf); [Hinton & Salakhutdinov, 2006](https://www.cs.toronto.edu/~hinton/science.pdf)).

Two ideas from that era never really left:

1. **Latents are causes.** A good generative model has a compressed internal state from which the image is rendered.
2. **Perception is inversion.** Recognition is approximately inverting that renderer.

VAEs, GANs, VQ-VAEs, and latent diffusion are different engineering answers to those two sentences.

## 2013–2014: two algorithms, two aesthetics

Modern deep generative modeling starts with a fork.

**Variational autoencoders** put a Gaussian (or similar) prior on a latent $$z$$, decode it to an image, and train by maximizing a lower bound on the log-likelihood — the ELBO ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114); [Rezende, Mohamed & Wierstra, 2014](https://arxiv.org/abs/1401.4082)). The math is clean. The samples, for years, were not. Averaging under a pixelwise likelihood makes models prefer blur.

**Generative adversarial networks** threw the likelihood out. A generator tries to fool a discriminator; at equilibrium, the generator samples from the data distribution ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)). Early GANs were unstable and low-resolution. They were also _sharp_. That aesthetic — crispy, slightly overconfident textures — defined a cultural era of AI images.

A parallel, quieter line kept exact likelihoods via invertible flows: NICE, then RealNVP, then Glow ([Dinh, Krueger & Bengio, 2014](https://arxiv.org/abs/1410.8516); [Dinh, Sohl-Dickstein & Bengio, 2016](https://arxiv.org/abs/1605.08803); [Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039)). Flows never won the photorealism contest. They kept an important property alive: you can evaluate $$p(x)$$, not only sample from it.

By 2015 the field had a personality split that lasted almost a decade:

- If you cared about **likelihood and latents**, you lived in VAE / flow / autoregressive land.
- If you cared about **looking real**, you lived in GAN land.

## 2015–2019: GANs learn to draw faces

The GAN engineering stack came together quickly. DCGANs showed that convolutional generators and discriminators could be trained at all ([Radford, Metz & Chintala, 2015](https://arxiv.org/abs/1511.06434)). Wasserstein GANs and gradient penalties made the loss less brittle ([Arjovsky, Chintala & Bottou, 2017](https://arxiv.org/abs/1701.07875); [Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)). Progressive growing, then StyleGAN, then StyleGAN2 turned unconditional face generation into a solved-looking demo ([Karras et al., 2017](https://arxiv.org/abs/1710.10196); [Karras, Laine & Aila, 2018](https://arxiv.org/abs/1812.04948); [Karras et al., 2019](https://arxiv.org/abs/1912.04958)). BigGAN scaled adversarial training to ImageNet class-conditional generation ([Brock, Donahue & Simonyan, 2018](https://arxiv.org/abs/1809.11096)).

Conditional image-to-image translation was a second GAN empire: pix2pix for paired data, CycleGAN for unpaired ([Isola et al., 2016](https://arxiv.org/abs/1611.07004); [Zhu et al., 2017](https://arxiv.org/abs/1703.10593)). For a while, "generative vision" _meant_ these papers.

Why didn't this stack become the foundation-model stack? In hindsight the reasons are boring and decisive:

- Adversarial training does not scale as politely as a denoising loss.
- Mode collapse and truncation tricks are ugly at internet scale.
- Text is a first-class conditioning signal, and GAN conditioning never became as simple as "attend to a language embedding in a noise predictor."

GANs taught the field that photorealism was possible. Diffusion inherited the ambition and discarded the game.

## The other camp: pixels as language

While GANs were winning Twitter, likelihood models were treating an image as a sequence.

PixelRNNs and PixelCNNs factorize $$p(x)$$ as a product of conditionals over pixels ([van den Oord, Kalchbrenner & Kavukcuoglu, 2016](https://arxiv.org/abs/1601.06759); [van den Oord et al., 2016](https://arxiv.org/abs/1606.05328)). This is slow. It is also the correct Bayesian story of an image as data. The important move was not to stay at pixels, but to _quantize_. VQ-VAE learned a discrete codebook so a second model could model codes instead of RGB ([van den Oord, Vinyals & Kavukcuoglu, 2017](https://arxiv.org/abs/1711.00937); [Razavi, van den Oord & Vinyals, 2019](https://arxiv.org/abs/1906.00446)). VQGAN put a perceptual / adversarial decoder on that idea and made discrete tokens good enough for high-resolution synthesis ([Esser, Rombach & Ommer, 2020](https://arxiv.org/abs/2012.09841)).

Once images are tokens, they look like language. iGPT trained a GPT-style transformer on pixels ([Chen et al., 2020](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)). DALL·E trained one on text tokens plus image tokens and showed that a large autoregressive transformer could do zero-shot text-to-image ([Ramesh et al., 2021](https://arxiv.org/abs/2102.12092)). That paper is easy to underrate now because the samples look dated. Architecturally it is the ancestor of every "image tokenizer + sequence model" system that followed.

CLIP, released the same season, is not a generator. It is the reason generators became _promptable_. Contrastive pretraining on image–text pairs produced a representation space where a caption is a location, not a class index ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)). Almost every visual foundation model since then is, in part, a way of sampling images that live near a CLIP (or CLIP-like) point.

## Diffusion: the sampler that scaled

The diffusion story starts earlier than the hype. Sohl-Dickstein et al. already described learning to reverse a gradual noising process in 2015 ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). It sat quietly until score matching and denoising autoencoders were re-read as generative models of the score $$\nabla_x \log p_t(x)$$ ([Song & Ermon, 2019](https://arxiv.org/abs/1907.05600)). Denoising diffusion probabilistic models made the discrete-time version simple enough to implement in a weekend ([Ho, Jain & Abbeel, 2020](https://arxiv.org/abs/2006.11239)). Score-SDE put the continuous-time picture in one framework ([Song et al., 2020](https://arxiv.org/abs/2011.13456)). DDIM showed you do not need the full stochastic chain to sample ([Song, Meng & Ermon, 2020](https://arxiv.org/abs/2010.02502)).

The political moment was 2021: a carefully ablated diffusion model beat GANs on ImageNet, with and without classifier guidance ([Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233); [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)). Classifier-free guidance then removed the need for a separate classifier ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)). That one trick — train with and without the condition, sample with an extrapolated condition — is still in essentially every production text-to-image system.

Then the engineering trick that made it a product: **do not diffuse pixels.** Latent diffusion runs the noising process in a VAE latent space, which is the VAE idea returning as infrastructure rather than as the generator itself ([Rombach et al., 2021](https://arxiv.org/abs/2112.10752)). GLIDE, DALL·E 2, Imagen, and Stable Diffusion landed within months of each other, with different bets on where language should live (text encoder vs. frozen LLM vs. CLIP prior) ([Nichol et al., 2021](https://arxiv.org/abs/2112.10741); [Ramesh et al., 2022](https://arxiv.org/abs/2204.06125); [Saharia et al., 2022](https://arxiv.org/abs/2205.11487)).

If you work on this daily, the conceptual shift is easy to miss. GANs _sampled_ in one forward pass and _fought_ a discriminator. Diffusion _samples_ by iterative refinement and _trains_ with a regression loss. Regression scales. Games, at this size, mostly do not.

## From a good sampler to a foundation model

"Foundation model" is a 2021 name for a 2018–2020 fact: some models are trained at enough scale, on enough heterogeneous data, that they become a substrate for many downstream tasks ([Bommasani et al., 2021](https://arxiv.org/abs/2108.07258)). For language that meant GPT and friends. For vision, classification backbones (ImageNet, then CLIP) were foundation models for _understanding_. Generative visual FMs are newer. A text-to-image model is not automatically one. It becomes one when the same prior is reused for editing, inpainting, variation, video, 3D, and — increasingly — as the visual half of a multimodal system that can both see and make.

A few technical moves turned T2I models into that substrate.

**Transformers as the denoiser.** U-Nets were the default backbone. Diffusion Transformers replaced the U-Net with a transformer over patches of the noisy latent ([Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748)). Once the denoiser is a transformer, scaling lore from LLMs starts to transfer.

**Flows, again, but simpler.** Flow matching and rectified flow train a network to transport noise to data along (approximately) straight paths, which is diffusion without some of the schedule archaeology ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747); [Liu, Gong & Liu, 2022](https://arxiv.org/abs/2209.03003)). Stable Diffusion 3 made that combination — rectified flow + a multimodal DiT — the open-source reference of the 2024 cohort ([Esser et al., 2024](https://arxiv.org/abs/2403.03206)). Flux and its descendants sit on the same branch, even when the lab note is a blog post rather than an arXiv id.

**Autoregression did not die.** It came back with better tokenizers and better factorization: next-scale prediction instead of next-pixel (VAR), continuous-token autoregression without a codebook (MAR), and large Llama-style image generators (LlamaGen) ([Tian et al., 2024](https://arxiv.org/abs/2404.02905); [Li et al., 2024](https://arxiv.org/abs/2406.11838); [Sun et al., 2024](https://arxiv.org/abs/2406.06525)). This matters for foundation models because a unified transformer that already speaks discrete tokens would rather generate images as tokens than host a separate diffusion head — unless, as Transfusion argues, you should just put both losses in one model ([Zhou et al., 2024](https://arxiv.org/abs/2408.11039)).

**Language and pixels in one network.** Chameleon tokenizes images and trains an early-fusion transformer on mixed sequences ([Chameleon Team, 2024](https://arxiv.org/abs/2405.09818)). Show-o and Janus-style models try to be both understanders and generators without two full stacks ([Xie et al., 2024](https://arxiv.org/abs/2408.12528); [Wu et al., 2024](https://arxiv.org/abs/2410.13848)). This is the visual analogue of "the model is the product": not a sampler you wrap with a prompt, but a multimodal prior you talk to.

**Time.** Images were the gym. Video is the actual sport. Stable Video Diffusion adapted image latents to temporal generation ([Blattmann et al., 2023](https://arxiv.org/abs/2311.15127)). VideoPoet treated video as a codec-token sequence modeled by a language model ([Kondratyuk et al., 2023](https://arxiv.org/abs/2312.14125)). Sora's report framed video models as world simulators rather than as T2I with extra frames ([Brooks et al., 2024](https://openai.com/index/video-generation-models-as-world-simulators/)). Whether that framing is true is an empirical question. That it is _sayable_ tells you the aspiration of current visual FMs: a prior over spatiotemporal reality, conditioned on language, usable as a tool.

## A compressed timeline

| Years     | Dominant object                           | What you actually train                           | What it was good at                  |
| --------- | ----------------------------------------- | ------------------------------------------------- | ------------------------------------ |
| 1995–2012 | Energy models, RBMs, DBNs                 | Contrastive / wake–sleep                          | Features, not pictures               |
| 2013–2014 | VAE, GAN, NICE                            | ELBO vs. a discriminator vs. a Jacobian           | Proofs of life                       |
| 2015–2019 | StyleGAN, BigGAN, PixelCNN, VQ-VAE        | Adversarial loss; pixel likelihood; codebooks     | Faces, translation, discrete latents |
| 2020–2021 | DDPM, CLIP, DALL·E, VQGAN                 | Denoising score; contrastive image–text; tokens   | The stack that still exists          |
| 2022–2023 | LDM, CFG, Imagen, DiT                     | Latent denoising + guidance                       | Text-to-image as a product           |
| 2024–2026 | MMDiT / flow, VAR/MAR, unified FMs, video | Flow matching; mixed AR + diffusion; joint tokens | Visual foundation models             |

The through-line is reuse. VAEs survived as latent compressors. Adversarial losses survived as decoders and perceptual critics. Autoregression survived as the interface to language. Diffusion / flow survived as the scalable way to sample continuous fields. Foundation models are what you get when those pieces are trained together, at a scale where the distinctions start to look like implementation details.

## What I think is still unsettled

A few arguments in this literature are not settled, and pretending they are is how survey posts go stale.

**Likelihood vs. perception.** FID, CLIP score, and Elo rankings on arena sites are not $$p(x)$$. We still do not have a single number that means "this is a better generative model." GANs taught us that a bad likelihood can look great. Diffusion taught us that a great sampler can still fail at spelling and counting. Visual FMs will be judged as _systems_ — edit, compose, remember, follow a script — more than as densities.

**Tokens vs. continuous fields.** Discrete codes play nicely with LLMs. Continuous latents play nicely with physics-like iterative refinement. Transfusion, MAR, and hybrid heads are evidence that this is a systems question, not a religious one.

**The world-model claim.** A video FM that respects object permanence, camera geometry, and causality would be a different object from a T2I model that happens to have a time axis. We do not yet have a clean experimental standard for that difference. Sora-style reports pointed at it; they did not close it.

**Data and compute.** Scaling stories for visual generators are less public than for LLMs, but the qualitative lesson is the same one Noam Shazeer had for language: FLOPs buy generalization, parameters buy memory. Caption quality, synthetic data, and filtering now move samples as much as architecture does.

## Closing

If you only remember one picture of this history, remember a relay rather than a replacement. Helmholtz machines asked for a generative model of sensation. VAEs made that variational. GANs made it sharp. Vector quantization made it discrete. CLIP made it linguistic. Diffusion made it scalable. Flow matching made the paths straighter. Transformers made the backbone familiar. Foundation models are the attempt to stop treating those as separate papers and start treating them as one prior over the visual world.

That prior is still incomplete. It is also, finally, good enough to be infrastructure. That is the real break from 2014. The interesting work is no longer proving that a net can draw. It is deciding what a visual foundation model is _for_.

## References

In-text links point at PDFs or arXiv abstracts. I list them here in roughly historical order so the bibliography is itself a timeline.

- Dayan, P., Hinton, G. E., Neal, R. M., & Zemel, R. S. (1995). The Helmholtz machine. _Neural Computation_. [PDF](https://www.cs.toronto.edu/~hinton/absps/helmholtz.pdf)
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. _Nature_. [paper](https://www.nature.com/articles/381607a0)
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. _Neural Computation_. [PDF](https://www.cs.toronto.edu/~hinton/absps/nccd.pdf)
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. _Neural Computation_. [PDF](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. _Science_. [PDF](https://www.cs.toronto.edu/~hinton/science.pdf)
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
- Goodfellow, I., et al. (2014). Generative adversarial nets. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear independent components estimation. [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)
- Sohl-Dickstein, J., et al. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel recurrent neural networks. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- van den Oord, A., et al. (2016). Conditional image generation with PixelCNN decoders. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
- Isola, P., et al. (2016). Image-to-image translation with conditional adversarial networks. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- Zhu, J.-Y., et al. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
- Gulrajani, I., et al. (2017). Improved training of Wasserstein GANs. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
- Karras, T., et al. (2017). Progressive growing of GANs for improved quality, stability, and variation. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1×1 convolutions. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- Brock, A., Donahue, J., & Simonyan, K. (2018). Large scale GAN training for high fidelity natural image synthesis. [arXiv:1809.11096](https://arxiv.org/abs/1809.11096)
- Karras, T., Laine, S., & Aila, T. (2018). A style-based generator architecture for generative adversarial networks. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
- Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating diverse high-fidelity images with VQ-VAE-2. [arXiv:1906.00446](https://arxiv.org/abs/1906.00446)
- Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- Karras, T., et al. (2019). Analyzing and improving the image quality of StyleGAN. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Chen, M., et al. (2020). Generative pretraining from pixels. ICML. [PDF](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
- Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Song, Y., et al. (2020). Score-based generative modeling through stochastic differential equations. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Esser, P., Rombach, R., & Ommer, B. (2020). Taming transformers for high-resolution image synthesis. [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)
- Ramesh, A., et al. (2021). Zero-shot text-to-image generation. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs on image synthesis. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
- Bommasani, R., et al. (2021). On the opportunities and risks of foundation models. [arXiv:2108.07258](https://arxiv.org/abs/2108.07258)
- Nichol, A., et al. (2021). GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. [arXiv:2112.10741](https://arxiv.org/abs/2112.10741)
- Rombach, R., et al. (2021). High-resolution image synthesis with latent diffusion models. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Ramesh, A., et al. (2022). Hierarchical text-conditional image generation with CLIP latents. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125)
- Saharia, C., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)
- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Lipman, Y., et al. (2022). Flow matching for generative modeling. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Peebles, W., & Xie, S. (2022). Scalable diffusion models with transformers. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- Blattmann, A., et al. (2023). Stable video diffusion: Scaling latent video diffusion models to large datasets. [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
- Kondratyuk, D., et al. (2023). VideoPoet: A large language model for zero-shot video generation. [arXiv:2312.14125](https://arxiv.org/abs/2312.14125)
- Esser, P., et al. (2024). Scaling rectified flow transformers for high-resolution image synthesis. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
- Tian, K., et al. (2024). Visual autoregressive modeling: Scalable image generation via next-scale prediction. [arXiv:2404.02905](https://arxiv.org/abs/2404.02905)
- Brooks, T., et al. (2024). Video generation models as world simulators. OpenAI. [report](https://openai.com/index/video-generation-models-as-world-simulators/)
- Chameleon Team. (2024). Chameleon: Mixed-modal early-fusion foundation models. [arXiv:2405.09818](https://arxiv.org/abs/2405.09818)
- Sun, P., et al. (2024). Autoregressive model beats diffusion: Llama for scalable image generation. [arXiv:2406.06525](https://arxiv.org/abs/2406.06525)
- Li, T., et al. (2024). Autoregressive image generation without vector quantization. [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)
- Xie, J., et al. (2024). Show-o: One single transformer to unify multimodal understanding and generation. [arXiv:2408.12528](https://arxiv.org/abs/2408.12528)
- Zhou, C., et al. (2024). Transfusion: Predict the next token and diffuse images with one multi-modal model. [arXiv:2408.11039](https://arxiv.org/abs/2408.11039)
- Wu, C., et al. (2024). Janus: Decoupling visual encoding for unified multimodal understanding and generation. [arXiv:2410.13848](https://arxiv.org/abs/2410.13848)
