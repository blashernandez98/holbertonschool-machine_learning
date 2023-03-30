
# Plotting intro project

## Making it work on WSL

- Install Xming Server
- Add this lines before importing matplotlib.pyplot:<br>
>import matplotlib
>
>matplotlib.use("TkAgg")