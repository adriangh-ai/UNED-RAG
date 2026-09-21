from pydantic import BaseModel, HttpUrl, Field, ValidationError, RootModel
from typing import List, Optional

class Answer(BaseModel):
    response_1: str = Field(..., alias="Response 1")
    response_2: str = Field(..., alias="Response 2")

class QuestionModel(BaseModel):
    id: int
    question: str = Field(..., alias="Question")
    localizar: bool = Field(..., alias="LOCALIZAR")
    answer: Answer = Field(..., alias="Answer")
    support_docs: List[HttpUrl] = Field(..., alias="Support Docs")
    support_docs_comments: Optional[str] = Field(None, alias="Support Docs Comments")
    support_text: List[str] = Field(..., alias="Support text")

class QuestionsListModel(RootModel[List[QuestionModel]]):
    pass