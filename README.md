# ATAI ChatBot
This is a class project form ATAI

## Environment Setup


1. Check if there is a Dataset fonder under project path, copy `14_graph.nt` to `Dataset/`
2. `cd <your-project-directory>`
3. `conda env create -f environment.yml`
4. Run `Application.py` and see if the sever can be setup

### Environment Update
Update `environment.yml` if a new dependency or package is installed
```
conda env update --file environment.yml
```
## Access ChatBot
1. Login to UZH [Speakeasy](https://speakeasy.ifi.uzh.ch/) with credentials
2. Select `start conversation`
3. type username (not the bot name, no idea why) and request to start chat
   1. This might deu to the fact that we are using speakeasy username to login instead of bot name. However I falied to login in with bot name and password.
   
