import pandas as pd
import sqlalchemy as sa
from fastapi_app.db import models


def get_max_project_id_of_user(user_id, db):
    subqry = db.query(sa.func.max(models.ProjectSetup.project_id)).filter(models.ProjectSetup.id == user_id)
    qry = db.query(models.ProjectSetup).filter(models.ProjectSetup.id == user_id,
                                               models.ProjectSetup.project_id == subqry)
    res = qry.first()
    max_project_id = res.project_id if hasattr(res, 'project_id') else None
    return max_project_id


def next_project_id_of_user(user_id, db):
    max_project_id = get_max_project_id_of_user(user_id, db)
    if pd.isna(max_project_id):
        next_project_id = 0
    else:
        next_project_id = max_project_id + 1
    return next_project_id