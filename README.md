# **_Overview_**

PyPty is a **phase retrieval** code that can be applied to **near-field or far-field imaging** in **TEM/STEM**. It can be applied to do **iterative ptychography**, **direct ptychography** (Wigner distribution deconsvolution), **differential phase contrast**, **tilt-corrected bright field**, **focal series reconstructions** and **LARBED reconstructions**.

The code is written by Anton Gladyshev (AG SEM, Physics Department, Humboldt-Universität zu Berlin). 




# **_Installation_**

## Setting Up the Python Environment and Installing PyPty

To create a proper Python environment and install PyPty, you can use **conda**, **mamba**, or **micromamba**. With **conda**, use:

### GPU Installation

```bash
git clone git@github.com:Anton-Gladyshev/pypty.git
cd pypty
conda env create -f pypty_gpu.yml
conda activate pypty
pip install .[gpu]
```

### CPU Installation

```bash
git clone git@github.com:Anton-Gladyshev/pypty.git
cd pypty
conda env create -f pypty_cpu.yml
conda activate pypty
pip install .
```


# **_Examples_**

The examples will be provided in the [examples folder](examples). To to configure a **completely custom reconstruction**, please reffer to the documentation.
 


# **_Documentation_**

Documentation website can be generated localy via 

```bash
bash generate_docs.sh
```

or accessed online https://anton-gladyshev.github.io/pypty


