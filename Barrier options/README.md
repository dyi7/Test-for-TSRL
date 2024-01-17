# Applying Lu(23)'s method to various Barrier options

In my final report, we show the pricing of:

  - **different barrier options**, i.e., Up-and-in, Up-and-out, Down-and-in, Down-and-out, using Lu(23)'s method in Section 3.1, 

  - **up-and-out put option with different 𝑋0** in Section 3.2, and 

  - **American up-and-out put option** in Section 3.3.


## How to run

- `script_barrier_up_out_put.py`: my application of Lu(23)'s method on **up-and-out PUT option**. You can change `barrier_direction='Up'` and `optionality = 'Out'` in the script to another barrier options such as 'up-and-in', 'down-and-in', 'down-and-out'.
- `script_barrier_up_out_call.py`: my application of Lu(23)'s method on **up-and-out CALL option**.
- `script_barrier_up_out_put_s0_0.6.py`: my application of Lu(23)'s method on **up-and-out PUT option with 𝑋0=0.6**. You can change `x_init = 0.6` to other values like `x_init = 0.8, 1.0, 1.2, 1.4`.
- `script_barrier_up_out_put_ameri.py`: my application of Lu(23)'s method on **American up-and-out PUT option**.

<br/>

For Deep solver method, run `script_barrier_up_out_call_deepsolver.py`. This part of code is modified from (https://github.com/AlessandroGnoatto/DeepBsdeSolverWithJumps). 

<br/>

The results we obtained and documented in the final report are also in <https://drive.google.com/drive/folders/1IA8Q2vOkdJKobUJM4M0PMPXxXaitD6T3?usp=drive_link.>


## Reference

[1] Lu, L., Guo, H., Yang, X., & Zhu, Y. (2023). Temporal difference learning for high-dimensional PIDEs with jumps. arXiv preprint arXiv:2307.02766.

[2] Gnoatto, A., Patacca, M., & Picarelli, A. (2022). A deep solver for BSDEs with jumps. arXiv preprint arXiv:2211.04349.
