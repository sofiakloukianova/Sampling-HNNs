<div align="center">
  <h1>
    Random Feature Hamiltonian Networks for N-Body Problems
  </h1>

  <h4>
    Bachelor's Thesis:
    <a href="thesis-docs/RF-HNNs-Thesis.pdf">RF-HNNs-Thesis.pdf</a><br>
    Bachelor's Thesis Presentation:
    <a href="thesis-docs/RF-HNNs-Presentation.pdf">RF-HNNs-Presentation.pdf</a>
  </h4>
</div>

---

<details open>
  <summary>
    <h2>Original Work</h2>
  </summary>

  This repository is based on the implementation from:
  <b>Training Hamiltonian Neural Networks without Backpropagation</b><br>
  - Repository: <a href="https://github.com/AlphaGergedan/Sampling-HNNs">https://github.com/AlphaGergedan/Sampling-HNNs</a><br>
  - Paper: <a href="https://arxiv.org/abs/2411.17511">https://arxiv.org/abs/2411.17511</a>
</details>

---

<details open>
  <summary>
    <h2>Models</h2>
  </summary>

  The following models are available:

  - <b>MLP</b>: ODE-Net, directly approximates <code>q_dot</code> and <code>p_dot</code>.
    <a href="https://arxiv.org/abs/1806.07366">paper</a>

  - <b>HNN</b>: Hamiltonian neural network approximates <code>H</code>, then recovers
    <code>q_dot</code> and <code>p_dot</code> using automatic differentiation and Hamilton’s equations.
    <a href="https://arxiv.org/abs/1906.01563v2">paper</a>

  All models are available in sampled form (<b>S-MLP</b>, <b>S-HNN</b>).
  In sampled models, hidden-layer parameters are randomly sampled and the final layer is computed
  using a least-squares solution.  
  Different sampling strategies are available through the SWIM method.
  <a href="https://arxiv.org/abs/2306.16830">paper</a>

  <div align="center">
    <img src="/assets/smlp-shnn.png" />
  </div>
</details>

---

<details>
  <summary>
    <h2>Setup</h2>
  </summary>

  ### Submodule
  This project depends on the submodule `swimnetworks`:
  ```sh
  git submodule init
  git submodule update
  ```
  to clone the submodule.

  ### Environment

  Create the conda environment:
  ```sh
  conda env create --file=environments.yml
  ```
  Then activate it with `conda activate s-hnn`.

  ### Examples
  After setting up the conda environment, you can use the bash script `main` located at the root of the
  project.
  - Run `./main --help` for usage.
  - Training a traditional network: `./main --target single_pendulum --model {MLP,HNN}`
  - Sampling a network: `./main --target single_pendulum --model {S-MLP,S-HNN}`

  Here is an example to quickly train a Sampled-HNN for single pendulum:
  ```sh
  python src/main.py --target single_pendulum --model S-HNN
  ```

  First-order error correction example:
  ```sh
  python src/main_limited_data.py --target single_pendulum --model S-HNN
  ```

  For details you can refer to the original <a href="https://arxiv.org/abs/2411.17511">paper</a>.
</details>

---

<details>
  <summary>
    <h2>This Project</h2>
  </summary>

  This repository extends the original Sampled-HNN framework to gravitational N-body systems.

  Main additions:
  - **2-body orbit generation** : `src/data/two_body_orbit_generator.py`
  - **3-body orbit generation** : `src/data/three_body_orbit_generator.py`
  - **Hamiltonian formulation of the 2-body system** : `src/hamiltonian/two_body.py`
  - **Hamiltonian formulation of the 3-body system** : `src/hamiltonian/three_body.py`
  - **Symplectic integrators** : `src/integrators/`
  - **Training and evaluation notebooks for 2-body and 3-body systems** : `analyze_two_body.ipynb` & `analyze_three_body.ipynb`
  - **Generated plots for the thesis**  `plots/`
</details>
