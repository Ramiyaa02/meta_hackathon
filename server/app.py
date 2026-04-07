import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app
import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()


