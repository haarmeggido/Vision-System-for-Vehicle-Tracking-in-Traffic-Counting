import cv2


def draw_line(frame, p1, p2, count_A_to_B, count_B_to_A):
    cv2.line(frame, p1, p2, (0, 255, 255), 2)
    label = f"A->B: {count_A_to_B} | B->A: {count_B_to_A}"
    cv2.putText(
        frame,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (10, 10, 255),
        2,
    )


def draw_track(frame, bbox, track_id: int, label: str, crossed: int = 0):
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"ID:{track_id}, label:{label}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    if crossed:
        cv2.putText(
            frame,
            f"CROSSED: {label}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (10, 255, 255),
            2,
        )