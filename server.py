import uvicorn

def main():
    uvicorn.run("deploy.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
