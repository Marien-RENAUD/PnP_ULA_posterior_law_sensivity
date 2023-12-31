# Plug-and-Play Unajusted Langevin Algorithm 

This code explore the PnP-ULA sensitivity to a denoiser shift and a measurement model shift. It is the code of the paper Plug-and-Play Posterior Sampling under Mismatched Measurement and Prior Models.

## Experiment for Gaussian Mixture Model in 2D

The folder pnp_ula_GMM_2D contains a notebook with the experiment for GMM in 2D. A parameter analysis in this toy example is also develop.

## Experiment on images

The folder pnp_ula_images contains the code use to run PnP-ULA on images (in gray-scale and in color). A specific README.md explain how to use the code in that folder. Various results of PnP-ULA on different images can also be find in the folder pnp_ula_images/results.

## Results

Here are some video presented samples during the stochastic process. Note how the quality increase during this process and how the Markov Chain continue to discover new possibilities.

<table>
  <tr>
    <td><img src="pnp_ula_images/results/result_gray/simpson_nb512/simpson_gif.gif" width="300" height="300" /></td>
    <td><img src="pnp_ula_images/results/result_rgb/woman02/woman_2_gif.gif" width="300" height="300" /></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="pnp_ula_images/results/result_rgb/woman03/woman_3_gif.gif" width="300" height="300" /></td>
    <td><img src="pnp_ula_images/results/result_rgb/castle/castle_gif.gif" width="300" height="300" /></td>
  </tr>
</table>

