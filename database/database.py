"""
Database Module for AI-Powered Recommendation System.
Supports SQLite (zero-config local) and MySQL (production).
"""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

def utc_now_naive():
    """Returns naive UTC datetime object avoiding Python 3.12+ utcnow deprecation."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Environment variable or default SQLite database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecommerce.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Set connect_args only for SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ========================================================
# SQLAlchemy ORM Models
# ========================================================

class Category(Base):
    __tablename__ = "categories"

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), nullable=False)
    icon = Column(String(50), default="tag")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String(50), primary_key=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100), nullable=False)
    brand = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    discount_percent = Column(Integer, default=0)
    popularity_score = Column(Float, default=0.0)
    stock = Column(Integer, default=100)
    image_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)

    # Relationships
    interactions = relationship("Interaction", back_populates="product", cascade="all, delete-orphan")
    ratings = relationship("Rating", back_populates="product", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    user_id = Column(String(50), primary_key=True)
    username = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    preferred_category = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)

    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    wishlist = relationship("Wishlist", back_populates="user", cascade="all, delete-orphan")


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    interaction_type = Column(String(20), nullable=False, index=True)  # view, click, like, wishlist, cart, purchase
    timestamp = Column(DateTime, default=utc_now_naive)

    user = relationship("User", back_populates="interactions")
    product = relationship("Product", back_populates="interactions")


class Rating(Base):
    __tablename__ = "ratings"

    rating_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    review_text = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now_naive)

    user = relationship("User", back_populates="ratings")
    product = relationship("Product", back_populates="ratings")


class Wishlist(Base):
    __tablename__ = "wishlist"

    wishlist_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    added_at = Column(DateTime, default=utc_now_naive)

    user = relationship("User", back_populates="wishlist")


class RecommendationLog(Base):
    __tablename__ = "recommendations"

    recommendation_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False)
    algorithm = Column(String(50), nullable=False)  # content, collaborative, hybrid, trending, similar, etc.
    score = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now_naive)


# ========================================================
# Helper Functions
# ========================================================

def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Context manager or dependency for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_interaction(db: Session, user_id: str, product_id: str, interaction_type: str) -> Interaction:
    """Logs a user interaction and updates product popularity in real time."""
    interaction = Interaction(
        user_id=user_id,
        product_id=product_id,
        interaction_type=interaction_type,
        timestamp=utc_now_naive()
    )
    db.add(interaction)

    # Increment product popularity score dynamically
    weights = {"view": 0.1, "click": 0.2, "like": 0.5, "wishlist": 0.8, "cart": 1.0, "purchase": 2.0}
    weight = weights.get(interaction_type, 0.1)

    product = db.query(Product).filter(Product.product_id == product_id).first()
    if product:
        product.popularity_score = round((product.popularity_score or 0.0) + weight, 2)

    db.commit()
    db.refresh(interaction)
    return interaction
