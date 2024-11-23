from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, Literal

@dataclass
class GameState:
    elapsed_time: Optional[float] = None          # 0
    player_states: List[Optional[int]] = None     # 1-8
    count_left: Optional[int] = None              # 9
    count_right: Optional[int] = None             # 10
    penalty_left: Optional[int] = None            # 11
    penalty_right: Optional[int] = None           # 12
    weapons: List[Optional[str]] = None           # 13-20
    stage: Optional[str] = None                   # 21
    asari_count: Optional[int] = None             # 22
    hoko_count: Optional[int] = None              # 23
    area_count: Optional[int] = None              # 24
    yagura_count: Optional[int] = None            # 25
    message: Optional[str] = None                 # 26
    player_detected: Optional[bool] = None        # 27
    timestamp: datetime = None                    # 29

    def __post_init__(self):
        if self.player_states is None:
            self.player_states = [None] * 8
        if self.weapons is None:
            self.weapons = [None] * 8

@dataclass
class DetectionMessage:
    type: Literal["game_state", "error", "stop"]
    payload: Union[GameState, str, None]

def create_game_state(data_list: List) -> DetectionMessage:
    """リストからGameStateを作成"""
    if len(data_list) != 33:
        return DetectionMessage(
            type="error", 
            payload=f"Invalid data length: expected 33, got {len(data_list)}"
        )
    
    try:
        game_state = GameState(
            elapsed_time=data_list[0],
            player_states=data_list[1:9],
            count_left=data_list[9],
            count_right=data_list[10],
            penalty_left=data_list[11],
            penalty_right=data_list[12],
            weapons=data_list[13:21],
            stage=data_list[21],
            asari_count=data_list[22],
            hoko_count=data_list[23],
            area_count=data_list[24],
            yagura_count=data_list[25],
            message=data_list[26],
            player_detected=data_list[27],
            timestamp=data_list[29]
        )
        return DetectionMessage(type="game_state", payload=game_state)
    except Exception as e:
        return DetectionMessage(
            type="error", 
            payload=f"Error creating GameState: {str(e)}"
        ) 