# ATAI ChatBot
This is a class project form ATAI

## Environment Setup

1. Download `speakeasy-python-client-library` from [here](https://lms.uzh.ch/auth/RepositoryEntry/17583866519/CourseNode/85421310449426/path%3D~~Tutorials~~00%5FSpeakeasy%5FProject/0)
2. Follow `Readme.md`
   1. 
   ```
   pip install [local]/[path]/[to]/[your]/speakeasy-python-client-library/dist/speakeasypy-1.0.0-py3-none-any.whl
   ```
      
      1. Check import `from speakeasypy import Speakeasy` see if it can be interpreted 
3. Create Dataset fonder under project path, copy `14_graph.ttl` to `Dataset/`
4. Run `Application.py` and see if the sever can be setup

## Access ChatBot
1. Login to UZH [Speakeasy](https://speakeasy.ifi.uzh.ch/) with credentials
2. Select `start conversation`
3. type username (not the bot name, no idea why) and request to start chat
   1. This might deu to the fact that we are using speakeasy username to login instead of bot name. However I falied to login in with bot name and password.
   
