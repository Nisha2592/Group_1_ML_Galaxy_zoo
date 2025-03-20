# Galaxy zoo classifier

## How to get the data (if connection with Leonardo cluster)

We want to get a subset of the data from the path using the `scp` command from the command line. At the same level of this `README.md` exist a file called `file_list.txt` that contains a random list of a constant number of images from the data folder in leonardo. In order to get the data you need to create a folder called `data/` and inside a folder called `images/`, should look like the following diagram:

```
./
├── README.md
├── file_list.txt
├── data/
│   └── images/
```

Then, you just need to run the following command (after login leonardo and have the certificate)

```bash
cat file_list.txt | xargs -I {} scp leonardo_alias:{} ./data/images
```

where `leonardo_alias` is the alias that you have on the `config` file inside `~/.ssh/` folder, i.e, the alias that you use to `ssh` leonardo. For getting the other 2 `.csv` files, you can execute the following commands:

```bash
scp leonardo_alias:/leonardo_work/ICT24_MHPC/data_projects/Project_1/data/zoo2MainSpecz.csv data
scp leonardo_alias:/leonardo_work/ICT24_MHPC/data_projects/Project_1/data/gz2_filename_mapping.csv data
```
