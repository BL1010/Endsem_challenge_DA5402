import requests 


def test_home(): 
    r = requests.get("http://localhost:8000/") 
    assert r.status_code == 200 
    
def test_recommend(): 
    r = requests.post(
        "http://localhost:8000/recommend",
        json = {"user":1, "k": 5}
    )
    assert r.status_code == 200 