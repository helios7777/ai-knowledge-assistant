from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Metric(Base):
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tool = Column(String(50))
    latency = Column(Float)
    confidence = Column(Float, nullable=True)
    query = Column(Text)
    result = Column(Text)
    tokens_used = Column(Integer, default=0)  

class MetricsDB:
    def __init__(self, db_path: str = "./monitoring.db"):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def log_metric(self, tool: str, latency: float, query: str, result: str, confidence: float = None):
        metric = Metric(
            tool=tool,
            latency=latency,
            confidence=confidence,
            query=query,
            result=result
        )
        self.session.add(metric)
        self.session.commit()
    
    def get_metrics(self, limit: int = 100):
        return self.session.query(Metric).order_by(Metric.timestamp.desc()).limit(limit).all()
    
    def get_avg_latency_by_tool(self):
        from sqlalchemy import func
        return self.session.query(
            Metric.tool,
            func.avg(Metric.latency).label('avg_latency')
        ).group_by(Metric.tool).all()

metrics_db = MetricsDB()