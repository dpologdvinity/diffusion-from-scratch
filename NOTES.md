# How it works

A walkthrough of the math in this repo, with pointers to where each piece lives in the code.

## 1. Forward process: adding noise (`ddpm/schedule.py`)

Pick a variance schedule β₁ … β_T (linear from 1e-4 to 0.02, T = 1000). Each step adds a little Gaussian noise:

    q(x_t | x_{t-1}) = N( sqrt(1 - β_t) · x_{t-1},  β_t · I )

With α_t = 1 − β_t and ᾱ_t = ∏_{s≤t} α_s, composing Gaussians gives a closed form for jumping straight from clean data to any noise level:

    q(x_t | x_0) = N( sqrt(ᾱ_t) · x_0,  (1 − ᾱ_t) · I )
    x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε,   ε ~ N(0, I)

That's `NoiseSchedule.q_sample`. With the linear schedule, ᾱ_T ≈ 4e-5, so x_T is essentially pure noise. `sqrt(1 − β_t)` shrinks the signal so the total variance stays at 1 instead of growing.

## 2. Training objective (`ddpm/train.py`)

The network ε_θ(x_t, t, y) is trained to predict the noise that was added:

    L = E_{x_0, y, t, ε} || ε − ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε,  t,  y ) ||²

**Why predict ε rather than x_0?** Ho et al. derive this "simple" loss from the variational bound, which is a weighted sum of per-timestep KL terms. Predicting ε reparameterizes the posterior mean, and dropping the per-t weights gives a loss that trains better in practice. ε also has unit variance at every t, which makes it an easy, well-scaled regression target. Predicting ε is equivalent to estimating the score: ∇ log p(x_t) = −ε / sqrt(1 − ᾱ_t).

Other training details:
- t is sampled uniformly. The output conv is zero-initialized so the model starts by predicting 0.
- An EMA (exponential moving average) of the weights is used for sampling, which makes samples noticeably more stable.
- **Label dropout for CFG:** with probability 0.1, the class label is replaced by a "null" label (index 10). The same network therefore learns both p(x | y) and p(x).

## 3. The network (`ddpm/unet.py`)

A UNet at 28 → 14 → 7 → 14 → 28 with skip connections. Each ResBlock adds a conditioning vector:

    emb = MLP(sinusoidal(t)) + Embedding(y)

The timestep tells the network how much noise to expect; the class embedding says what to draw. Self-attention runs only at 7×7 (49 tokens), where it is cheap. GroupNorm is used instead of BatchNorm because batch statistics would mix very different noise levels.

## 4. DDPM sampling (`ddpm_step` in `ddpm/sampling.py`)

Start from x_T ~ N(0, I) and step backwards with the true posterior given a predicted clean image:

    x̂_0 = (x_t − sqrt(1 − ᾱ_t) · ε_θ) / sqrt(ᾱ_t)          (clipped to [-1, 1])
    q(x_{t-1} | x_t, x̂_0) = N( μ̃_t,  β̃_t I )
    μ̃_t = sqrt(ᾱ_{t-1}) β_t / (1 − ᾱ_t) · x̂_0  +  sqrt(α_t)(1 − ᾱ_{t-1}) / (1 − ᾱ_t) · x_t
    β̃_t = (1 − ᾱ_{t-1}) / (1 − ᾱ_t) · β_t

That's 1000 network evaluations per sample, because the Gaussian-step approximation only holds when each step is small.

## 5. DDIM sampling (`ddim_step`)

Song et al. observed that the training loss only depends on the marginals q(x_t | x_0), not on the step-by-step Markov chain. So we can pick a *different*, non-Markovian reverse process with the same marginals:

    x_{t'} = sqrt(ᾱ_{t'}) · x̂_0  +  sqrt(1 − ᾱ_{t'} − σ²) · ε̂  +  σ · z
    σ = η · sqrt( (1 − ᾱ_{t'}) / (1 − ᾱ_t) ) · sqrt( 1 − ᾱ_t / ᾱ_{t'} )

- The update only needs ᾱ at the two endpoints, so t' can be **any earlier timestep**. We run on an evenly spaced subsequence, for example 50 of the 1000 steps. **No retraining is needed**: the same ε_θ is used.
- **η = 0:** deterministic. The same x_T always gives the same image, and it behaves like an ODE solver (the probability-flow ODE), which is why large steps stay accurate.
- **η = 1** with t' = t − 1 is exactly the DDPM step. `tests/test_diffusion.py::test_ddim_with_eta_one_equals_ddpm_step` checks this numerically, to 1e-10 in float64.

Intuition: DDPM injects fresh noise every step, so it needs many small steps to average out the errors. DDIM with η = 0 follows a smooth deterministic path, so it can take large steps.

**Measured here (MNIST, N = 1,000, w = 2):** η, not the step count, explains most of the gap. At a fixed 50 steps, FID goes 19.0 (η = 0) → 22.6 (η = 0.5) → 30.1 (η = 1), close to DDPM 1000's 28.0. At 10 steps, η = 1 nearly doubles FID (36.4 vs 18.6): each large step injects noise that the few remaining steps cannot remove. FashionMNIST shows the same ordering. So DDIM's 50-step result is not "DDPM, but faster": it samples a different, deterministic path that happens to score better under this metric.

## 6. Classifier-free guidance (`guided_eps`)

At each sampling step, run the network twice (batched into one call), with the real label and with the null label:

    ε̃ = ε_θ(x_t, t, ∅) + w · ( ε_θ(x_t, t, y) − ε_θ(x_t, t, ∅) )

- w = 0 → unconditional, w = 1 → plain conditional, w > 1 → extrapolate past the conditional prediction.
- **Why it works:** ε relates to the score, and Bayes' rule gives ∇ log p(y | x) = ∇ log p(x | y) − ∇ log p(x). So the difference term is an implicit classifier gradient, and w scales it up. That corresponds to sampling from roughly p(x) · p(y | x)^w, which sharpens class identity.
- **Trade-off:** higher w gives more on-class, cleaner samples but less diversity. Past some w, FID gets worse even as class accuracy keeps rising. See `results/*_guidance_tradeoff.png`.
- "Classifier-free" means no separate noisy-image classifier is trained (unlike Dhariwal & Nichol's classifier guidance). The guidance signal comes from the same network via label dropout.

## 7. Evaluation (`ddpm/evaluate.py`, `ddpm/classifier.py`, `ddpm/metrics.py`)

- **Class accuracy:** a small CNN trained on the real dataset classifies each generated image; the score is the fraction matching the requested label.
- **FID:** fit a Gaussian to the features of generated images and another to real test images, then take the Fréchet distance between them: ‖μ₁ − μ₂‖² + Tr(Σ₁ + Σ₂ − 2(Σ₁Σ₂)^{1/2}). Standard FID uses ImageNet Inception features, which suit 28×28 grayscale digits poorly, so this repo uses the classifier's 64-d features. **The numbers are only comparable within this repo, not to published FIDs.** FID is biased at small sample sizes, so every config uses the same N, and "real images" with the same N gives the floor.
- **KID:** an unbiased MMD² with a cubic polynomial kernel on the same features. It's more reliable than FID with only a few hundred samples.
- **Speed:** reported as sampling steps (network calls; with CFG each call has a doubled batch) and as measured wall-clock time.

## References

- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* 2020.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* 2021.
- Ho, Salimans. *Classifier-Free Diffusion Guidance.* 2022.
- Nichol, Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* 2021 (cosine schedule).
- Bińkowski et al. *Demystifying MMD GANs.* 2018 (KID).
