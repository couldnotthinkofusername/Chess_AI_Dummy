import streamlit as st
import chess
import chess.svg
import numpy as np
import torch
import torch.nn
from sklearn.linear_model import LinearRegression
from streamlit_extras.let_it_rain import rain
 


def evaluate_board(board: chess.Board) -> float:

    material = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    PST_PAWN = np.array([
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ])
    PST_KNIGHT = np.array([
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ])
    PST_BISHOP = np.array([
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ])
    PST_ROOK = np.array([
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    ])
    PST_QUEEN = np.array([
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ])
    PST_KING = np.array([
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ])

    pst_by_piece = {
        chess.PAWN: PST_PAWN,
        chess.KNIGHT: PST_KNIGHT,
        chess.BISHOP: PST_BISHOP,
        chess.ROOK: PST_ROOK,
        chess.QUEEN: PST_QUEEN,
        chess.KING: PST_KING,
    }

    score = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            base = material[piece.piece_type]
            pst = pst_by_piece[piece.piece_type]
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            val = base + pst[idx]
            score += val if piece.color == chess.WHITE else -val


    b_w = board.copy()
    b_w.turn = chess.WHITE
    b_b = board.copy()
    b_b.turn = chess.BLACK
    mobility = (len(list(b_w.legal_moves)) - len(list(b_b.legal_moves))) * 2.0
    score += mobility


    try:
        if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
            score += 15
        if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
            score -= 15
    except Exception:
        pass

    return score / 100.0 


def extract_features(board: chess.Board) -> np.ndarray:
    features = np.zeros(12)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = (piece.piece_type - 1) + 6 * (1 if piece.color == chess.BLACK else 0)
            features[idx] += 1
    return features.reshape(1, -1)


def train_simple_model() -> LinearRegression:
    X = np.random.rand(100, 12) * 10
    y = np.sum(X[:, :6], axis=1) - np.sum(X[:, 6:], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    return model


class SimpleNN(torch.nn.Module):
    def __init__(self) -> None:
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_torch_model(model, epochs=100, lr=0.01):
    
    return model



def evaluate_combined(board: chess.Board, sklearn_model: LinearRegression, torch_model: SimpleNN) -> float:
   
    return float(evaluate_board(board))


# Minimax with alpha-beta pruning
def minimax_best_move(board: chess.Board, depth: int, sklearn_model: LinearRegression, torch_model: SimpleNN, ai_color: bool) -> chess.Move | None:
    def search(node: chess.Board, d: int, alpha: float, beta: float) -> float:
        if d == 0 or node.is_game_over():
            score = evaluate_combined(node, sklearn_model, torch_model)
            return score if ai_color == chess.WHITE else -score

        legal = list(node.legal_moves)
        if not legal:
            score = evaluate_combined(node, sklearn_model, torch_model)
            return score if ai_color == chess.WHITE else -score

        maximizing = (node.turn == ai_color)
        best_val = -float('inf') if maximizing else float('inf')

        def move_key(m: chess.Move):
            return 0 if node.is_capture(m) else 1
        legal.sort(key=move_key)

        for mv in legal:
            node.push(mv)
            val = search(node, d - 1, alpha, beta)
            node.pop()
            if maximizing:
                if val > best_val:
                    best_val = val
                if best_val > alpha:
                    alpha = best_val
                if alpha >= beta:
                    break
            else:
                if val < best_val:
                    best_val = val
                if best_val < beta:
                    beta = best_val
                if alpha >= beta:
                    break
        return best_val

    best_move = None
    best_score = -float('inf')
    moves = list(board.legal_moves)
    if not moves:
        return None

    def top_key(m: chess.Move):
        return 0 if board.is_capture(m) else 1
    moves.sort(key=top_key)

    for mv in moves:
        board.push(mv)
        score = search(board, depth - 1, -float('inf'), float('inf'))
        board.pop()
        if score > best_score:
            best_score = score
            best_move = mv
    return best_move





def ai_move(board: chess.Board, sklearn_model: LinearRegression, torch_model: SimpleNN):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = None
    best_score = -float('inf')

    for move in legal_moves:
        board.push(move)

        basic_score = evaluate_board(board)

        features = extract_features(board)
        sklearn_score = sklearn_model.predict(features)[0]

        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            torch_score = torch_model(features_tensor).item()

        combined_score = (basic_score + sklearn_score + torch_score) / 3
        if combined_score > best_score:
            best_score = combined_score
            best_move = move

        board.pop()

    return best_move


def main() -> None:
    st.set_page_config(page_title=" Chess AI Dummy", page_icon="♟️", layout="centered")
    st.markdown(
        """
        <style>
        .title-box {background: linear-gradient(90deg,#00c6ff 0%,#0072ff 100%); padding: 14px 18px; border-radius: 12px; color: #fff; margin-bottom: 14px;}
        .badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600;}
        .badge-turn {background:#ffe08a; color:#5a3e00;}
        .badge-eval-pos {background:#b6f3c0; color:#0f5132;}
        .badge-eval-neg {background:#ffb3b3; color:#58151c;}
        .panel {background: #ffffff; border: 1px solid #ececec; border-radius: 12px; padding: 12px 14px;}
        .move-list {max-height: 320px; overflow-y: auto; line-height: 1.7;}
        hr {border: none; height: 1px; background: linear-gradient(90deg, rgba(0,0,0,0) 0%, #e6e6e6 50%, rgba(0,0,0,0) 100%);} 
        .spacer-sm {height: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='title-box'><h2 style='margin:0'>Chess AI Dummy (VIBE CODED)</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    if 'board' not in st.session_state:
        st.session_state.board = chess.Board()
    if 'player_color' not in st.session_state:
        st.session_state.player_color = chess.WHITE
    if 'sklearn_model' not in st.session_state:
        st.session_state.sklearn_model = train_simple_model()
    if 'torch_model' not in st.session_state:
        st.session_state.torch_model = SimpleNN()
    if 'move_input' not in st.session_state:
        st.session_state.move_input = ""
    if 'clear_move' not in st.session_state:
        st.session_state.clear_move = False
    if 'ai_check' not in st.session_state:
        st.session_state.ai_check = False
    if 'ai_check_san' not in st.session_state:
        st.session_state.ai_check_san = ""
    board = st.session_state.board
    player_color = st.session_state.player_color

    
    depth = 4

    # Right panel helpers
    def get_san_history(b: chess.Board):
        temp = chess.Board()
        sans = []
        for mv in b.move_stack:
            sans.append(temp.san(mv))
            temp.push(mv)
        return sans

    cur_eval = evaluate_combined(board, st.session_state.sklearn_model, st.session_state.torch_model)
    eval_badge = f"<span class='badge {'badge-eval-pos' if cur_eval>=0 else 'badge-eval-neg'}'>Eval: {cur_eval:+.2f}</span>"
    turn_badge = f"<span class='badge badge-turn'>{'Your' if board.turn==player_color else 'AI'} turn</span>"

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown(f"{turn_badge} &nbsp; {eval_badge}", unsafe_allow_html=True)
        svg = chess.svg.board(board=board, size=460)
        st.markdown(svg, unsafe_allow_html=True)
        # Label with inline info popover next to it
        lbl_col, info_col = st.columns([1, 0.15])
        with lbl_col:
            st.write("Enter Move (SAN)")
        with info_col:
            try:
                with st.popover("ℹ️"):
                    st.markdown(
                        """
                        Use notation such as: e4 or Nf3.

                        Use O-O for kingside castle, O-O-O for queenside castle.

                        For pawn captures, write from-square then to-square (e.g., c4b5).
                        """
                    )
            except Exception:
                with st.expander("ℹ️ Move input help"):
                    st.markdown(
                        """
                        Use notation such as: e4 or Nf3.

                        Use O-O for kingside castle, O-O-O for queenside castle.

                        For pawn captures, write from-square then to-square (e.g., c4b5).
                        """
                    )
     
        if st.session_state.clear_move:
            st.session_state.move_input = ""
            st.session_state.clear_move = False
        move_input = st.text_input("Your move:", key="move_input")
        # Show AI Check notification just below the input
        if st.session_state.ai_check:
            st.markdown(
                "<div style='background:#ffe3e3;border:1px solid #ffb3b3;"
                "padding:8px 12px;border-radius:10px;color:#7a1212;display:inline-block;"
                "margin-top:6px;'>"
                + f"♚ Check! ({st.session_state.ai_check_san})" + "</div>",
                unsafe_allow_html=True,
            )
            # reset flag after surfacing once
            st.session_state.ai_check = False
            st.session_state.ai_check_san = ""

    if board.is_game_over():
        st.markdown("<h2 style='color:#FF5733;'>🎉 Game over! 🎉</h2>", unsafe_allow_html=True)
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            if winner == "White":
                st.markdown(
                    "<div style='background: linear-gradient(90deg, #f8ffae 0%, #43c6ac 100%);"
                    "padding: 1em; border-radius: 10px; color: #222; font-size: 1.2em;'>"
                    "♔ <b>Checkmate! White wins.</b> <br>Great job if you played as White!<br>"
                    "If not, try to spot where you could have improved your defense."
                    "</div>", unsafe_allow_html=True)
                rain(
                    emoji="♔",
                    font_size=54,
                    falling_speed=5,
                    animation_length="infinite"
                )
            else:
                st.markdown(
                    "<div style='background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);"
                    "padding: 1em; border-radius: 10px; color: #222; font-size: 1.2em;'>"
                    "♚ <b>Checkmate! Black wins.</b> <br>If you played as Black, congratulations!<br>"
                    "If not, review the game to see how you can avoid checkmate next time."
                    "</div>", unsafe_allow_html=True)
                rain(
                    emoji="♚",
                    font_size=54,
                    falling_speed=5,
                    animation_length="infinite"
                )
        elif board.is_stalemate():
            st.markdown(
                "<div style='background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);"
                "padding: 1em; border-radius: 10px; color: #333; font-size: 1.2em;'>"
                "🤝 <b>Stalemate!</b> <br>It's a draw. Both sides played well and neither could force a win.<br>"
                "Try to find a way to break the deadlock in your next game!"
                "</div>", unsafe_allow_html=True)
            rain(
                emoji="🤝",
                font_size=44,
                falling_speed=4,
                animation_length="infinite"
            )
        else:
            st.markdown(
                "<div style='background: linear-gradient(90deg, #e0c3fc 0%, #8ec5fc 100%);"
                "padding: 1em; border-radius: 10px; color: #333; font-size: 1.2em;'>"
                "🤷 <b>Draw!</b> <br>The game ended in a draw. Sometimes, a draw is the best result for both players.<br>"
                "Analyze the final position to see if either side missed a win."
                "</div>", unsafe_allow_html=True)
            rain(
                emoji="🤷",
                font_size=44,
                falling_speed=4,
                animation_length="infinite"
            )
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Reset Game"):
            st.session_state.board = chess.Board()
            st.rerun()
        return

    with right:
        sans = get_san_history(board)
        rows = []
        for i in range(0, len(sans), 2):
            white_mv = sans[i]
            black_mv = sans[i+1] if i+1 < len(sans) else ""
            if black_mv:
                row = f"<span style='color:#000000'>{white_mv} - {black_mv}</span>"
            else:
                row = f"<span style='color:#000000'>{white_mv}</span>"
            rows.append(row)
        history_html = "".join([f"<div>{row}</div>" for row in rows])
        st.markdown(
            "<div class='panel'>"
            "<b style='color:#000000'>Move history</b>"
            "<div class='move-list'>" + history_html + "</div>"
            "</div>",
            unsafe_allow_html=True
        )

    if board.turn == player_color:
        if move_input:
            try:
                move = board.parse_san(move_input.strip())
                if move in board.legal_moves:
                    board.push(move)
                    st.session_state.board = board
                    st.session_state.clear_move = True
                    st.rerun()
                else:
                    st.error("Illegal move!")
                    st.session_state.clear_move = True
            except ValueError:
                st.error("Invalid move format!")
                st.session_state.clear_move = True
    else:
        with left:
            st.write("AI is thinking...")
        ai_color = not player_color
        move = minimax_best_move(board, depth, st.session_state.sklearn_model, st.session_state.torch_model, ai_color)
        if move is None:
            move = ai_move(board, st.session_state.sklearn_model, st.session_state.torch_model)
        if move:
          
            san_text = board.san(move)
            board.push(move)
            st.session_state.board = board
            st.session_state.clear_move = True
            if board.is_check():
                st.session_state.ai_check = True
                st.session_state.ai_check_san = san_text
            st.rerun()
        else:
            with left:
                st.write("No legal moves for AI.")


if __name__ == "__main__":
    main()

