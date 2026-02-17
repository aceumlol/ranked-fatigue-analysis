"""
SQLAlchemy ORM models for the ranked fatigue analysis
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Numeric,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Player(Base):
    """Tracked high-elo players"""
    __tablename__ = "players"
    
    player_id = Column(String(100), primary_key=True)
    puuid = Column(String(150), unique=True, nullable=False, index=True)
    summoner_name = Column(String(100))
    region = Column(String(10), nullable=False)
    rank_tier = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    engineered_features = relationship("EngineeredFeature", back_populates="player")
    performance_metrics = relationship("PerformanceMetric", back_populates="player")
    
    def __repr__(self):
        return f"<Player(summoner_name='{self.summoner_name}', region='{self.region}')>"

class Match(Base):
    """Individual matches"""
    __tablename__ = "matches"
    
    match_id = Column(String(50), primary_key=True)
    region = Column(String(10), nullable=False)
    game_creation = Column(DateTime(timezone=True), nullable=False, index=True)
    game_duration = Column(Integer, nullable=False)
    game_version = Column(String(20), index=True)
    queue_id = Column(Integer)
    game_mode = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    participants = relationship("MatchParticipant", back_populates="match")
    
    def __repr__(self):
        return f"<Match(match_id='{self.match_id}', game_creation='{self.game_creation}')>"

class MatchParticipant(Base):
    """Bridge table: player performance in a specific match"""
    __tablename__ = "match_participants"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False)
    player_id = Column(String(100), nullable=False)
    team_id = Column(Integer, nullable=False)
    champion_id = Column(Integer, nullable=False, index=True)
    champion_name = Column(String(50))
    role = Column(String(20), index=True)
    lane = Column(String(20))
    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    kda = Column(Numeric(5, 2))
    gold_earned = Column(Integer)
    total_minions_killed = Column(Integer)
    cs_per_min = Column(Numeric(5, 2))
    gold_diff_15 = Column(Integer)
    total_damage_dealt_to_champions = Column(Integer)
    damage_share = Column(Numeric(5, 4))
    gold_share = Column(Numeric(5, 4))
    death_share = Column(Numeric(5, 4))
    vision_score = Column(Integer)
    wards_placed = Column(Integer)
    wards_killed = Column(Integer)
    win = Column(Boolean, nullable=False)
    match = relationship("Match", back_populates="participants")

    __table_args__ = (
        UniqueConstraint('match_id', 'player_id', name='uq_match_player'),
        Index('idx_participant_player', 'player_id'),
        Index('idx_participant_match', 'match_id'),
    )
    
    def __repr__(self):
        return f"<MatchParticipant(match_id='{self.match_id}', player_id='{self.player_id}', champion='{self.champion_name}')>"

class EngineeredFeature(Base):
    """Features for each match-player combo"""
    __tablename__ = "engineered_features"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey("matches.match_id"), nullable=False)
    player_id = Column(String(100), ForeignKey("players.player_id"), nullable=False)
    games_last_24h = Column(Integer)
    games_last_72h = Column(Integer)
    rest_hours_since_last_game = Column(Numeric(6, 2))
    rolling_kda_7 = Column(Numeric(5, 2))
    rolling_kda_14 = Column(Numeric(5, 2))
    rolling_deaths_7 = Column(Numeric(5, 2))
    rolling_gold_diff_7 = Column(Numeric(8, 2))
    rolling_std_kda_7 = Column(Numeric(5, 2))
    rolling_std_gold_diff_7 = Column(Numeric(8, 2))
    current_win_streak = Column(Integer)
    current_loss_streak = Column(Integer)
    fatigue_score = Column(Numeric(4, 3), index=True)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    is_weekend = Column(Boolean)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    player = relationship("Player", back_populates="engineered_features")

    __table_args__ = (
        UniqueConstraint('match_id', 'player_id', name='uq_feature_match_player'),
        Index('idx_feature_player', 'player_id'),
    )
    
    def __repr__(self):
        return f"<EngineeredFeature(player_id='{self.player_id}', fatigue_score={self.fatigue_score})>"

class PerformanceMetric(Base):
    """Pre-aggregated daily performance metrics per player"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(100), ForeignKey("players.player_id"), nullable=False)
    analysis_date = Column(DateTime(timezone=True), nullable=False)
    total_games = Column(Integer)
    win_rate = Column(Numeric(5, 4))
    avg_kda = Column(Numeric(5, 2))
    avg_deaths = Column(Numeric(5, 2))
    avg_gold_diff_15 = Column(Numeric(8, 2))
    avg_cs_per_min = Column(Numeric(5, 2))
    std_kda = Column(Numeric(5, 2))
    std_deaths = Column(Numeric(5, 2))
    cv_performance = Column(Numeric(5, 4))
    high_volatility_flag = Column(Boolean)
    performance_drop_flag = Column(Boolean)
    player = relationship("Player", back_populates="performance_metrics")

    __table_args__ = (
        UniqueConstraint('player_id', 'analysis_date', name='uq_player_date'),
        Index('idx_metrics_player_date', 'player_id', 'analysis_date'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(player_id='{self.player_id}', date='{self.analysis_date}')>"