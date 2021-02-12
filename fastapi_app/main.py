from fastapi.param_functions import Query
import models
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import SessionLocal, engine
from pydantic import BaseModel
from models import Nodes
from sqlalchemy.orm import Session
from typing import Optional
import sqlite3


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")


class AddNodeRequest(BaseModel):
    latitude: float
    longitude: float
    node_type: str
    fixed_type: bool


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    nodes = db.query(Nodes)
    return templates.TemplateResponse("home.html", {
        "request": request, "nodes": nodes
    })


@app.get("/nodes_db_html")
async def get_nodes(request: Request, db: Session = Depends(get_db)):
    res = db.execute("select * from nodes")
    result = res.fetchall()
    return result


def compute_links():
    pass


@app.post("/add_node/")
async def add_node(add_node_request: AddNodeRequest,
                   background_tasks: BackgroundTasks,
                   db: Session = Depends(get_db)):
    nodes = Nodes()

    nodes.latitude = add_node_request.latitude
    nodes.longitude = add_node_request.longitude
    nodes.node_type = add_node_request.node_type
    nodes.fixed_type = add_node_request.fixed_type

    db.add(nodes)
    db.commit()

    background_tasks.add_task(compute_links)

    return {
        "code": "success",
        "message": "node added to db"
    }


@app.post("/clear_node_db/")
async def add_node():

    sqliteConnection = sqlite3.connect('nodes.db')
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from nodes"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()

    return {
        "code": "success",
        "message": "nodes cleared"
    }
