import re
import json
import csv
from pathlib import Path

from bs4 import BeautifulSoup


# このスクリプトファイルと同じディレクトリをシラバスフォルダとして扱う
try:
    SYLLABUS_DIR = Path(__file__).resolve().parent
except NameError:
    # Jupyter 等で __file__ が無い場合のフォールバック
    SYLLABUS_DIR = Path(".").resolve()

# CSV に出す列の順番（対象学年は削除）
FIELD_ORDER = [
    "科目名",
    "開講学期",
    "曜日・時限",
    "単位数",
    "授業形態",
    "実施方法",
    "教授名",
    "到達目標",
    "概要",
    "成績評価基準",
    "成績評価結果",
]


def parse_course_block(soup: BeautifulSoup) -> dict:
    """
    科目名 / 開講学期 / 曜日・時限 / 単位数 / 授業形態 / 実施方法 / 教授名 をまとめて取る。
    同志社シラバス HTML の構造に合わせた実装。
    """
    result = {
        "科目名": None,
        "開講学期": None,
        "曜日・時限": None,
        "単位数": None,
        "授業形態": None,
        "実施方法": None,  # 面接/Face-to-face, Online など
        "教授名": None,
    }

    # --- 曜日・時限 + 実施方法（例：2024年度 (金曜日1講時) 面接/Face-to-face） ---
    header_table = soup.find("table", class_="show__content")
    if header_table:
        p_right = header_table.find("p", style=lambda v: v and "float:right" in v)
        if p_right:
            header_text = p_right.get_text(" ", strip=True)

            # (金曜日1講時) の中身 → 曜日・時限
            m = re.search(r"\(([^)]+)\)", header_text)
            if m:
                result["曜日・時限"] = m.group(1)

            # カッコ以降の部分 → 実施方法（面接/Face-to-face など）
            # 例: "2024年度 (金曜日1講時) 面接/Face-to-face"
            m2 = re.search(r"\)\s*(.+)$", header_text)
            if m2:
                method_text = m2.group(1).strip()
                if method_text:
                    result["実施方法"] = method_text

    # --- 科目名・単位数・開講学期・授業形態 ---
    # 「2単位/Unit 秋学期/Fall 京田辺/Kyotanabe 講義/Lecture」が入っている td
    course_info_td = None
    for td in soup.select("td.show__content-in"):
        if "単位/Unit" in td.get_text():
            course_info_td = td
            break

    if course_info_td:
        # 科目名：色付き font の中のテキスト（先頭の記号 ○/△ などを削除）
        font_tag = course_info_td.find("font")
        if font_tag:
            name = font_tag.get_text(strip=True)
            # 先頭の丸印などを削る
            name = re.sub(r"^[○△●◎◇◆☆★\s]+", "", name)
            result["科目名"] = name

        # 単位/学期/授業形態： '単位/Unit' を含む <p> を探してそこから抜く
        info_p = None
        for p in course_info_td.find_all("p"):
            if "単位/Unit" in p.get_text():
                info_p = p
                break

        if info_p:
            info_text = info_p.get_text(" ", strip=True)

            # 単位数（例：2単位/Unit）
            m = re.search(r"(\d+)単位", info_text)
            if m:
                try:
                    result["単位数"] = int(m.group(1))
                except ValueError:
                    result["単位数"] = m.group(1)

            # 開講学期（春学期/秋学期/前期/後期/通年 など）
            m = re.search(r"(春学期|秋学期|前期|後期|通年)", info_text)
            if m:
                result["開講学期"] = m.group(1)

            # 授業形態（講義/演習/実験/実習…）
            m = re.search(r"(講義|演習|実験|実習|ゼミナール|講義・演習|講義・実習)", info_text)
            if m:
                result["授業形態"] = m.group(1)

    # --- 教授名（教員プロフィールへのリンクから） ---
    prof_link = soup.select_one('td.show__content-in a[href*="kendb.doshisha.ac.jp/profile"]')
    if prof_link:
        result["教授名"] = prof_link.get_text(strip=True)

    return result


def extract_section_text(soup: BeautifulSoup, label_keyword: str):
    """
    ＜概要/..., ＜到達目標/..., などの見出し b タグの次の <p> をテキストで返す。
    label_keyword には "＜概要" や "＜到達目標" などを渡す。
    """
    b = soup.find("b", string=lambda s: s and label_keyword in s)
    if not b:
        return None

    p_label = b.find_parent("p")
    if not p_label:
        return None

    content_p = p_label.find_next_sibling("p")
    if not content_p:
        return None

    return " ".join(content_p.stripped_strings)


def parse_evaluation_criteria(soup: BeautifulSoup):
    """
    ＜成績評価基準/Evaluation Criteria＞の表 (class="show__grades") を
    [ {項目, 割合, 詳細}, ... ] のリストにする。
    何も取れなければ None。
    """
    table = soup.find("table", class_="show__grades")
    if not table:
        return None

    rows = table.find_all("tr")
    items = []
    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) != 3:
            continue
        item = {
            "項目": tds[0].get_text(strip=True),
            "割合": tds[1].get_text(strip=True),
            "詳細": tds[2].get_text(strip=True),
        }
        # 完全に空の行は飛ばす
        if any(item.values()):
            items.append(item)

    return items or None


def parse_grade_results(soup: BeautifulSoup):
    """
    ＜成績評価結果/Results of assessment＞の表を
    { '登録者数': ..., 'A': ..., ..., '備考': ... } の dict にする。
    何も取れなければ None。
    """
    b = soup.find("b", string=lambda s: s and "＜成績評価結果" in s)
    if not b:
        return None

    p_label = b.find_parent("p")
    if not p_label:
        return None

    table = p_label.find_next("table")
    if not table:
        return None

    rows = table.find_all("tr")
    if len(rows) < 3:
        return None

    # 3 行目がデータ行（登録者数, A, B, C, D, F, 他, 評点平均値, 備考）
    data_tds = rows[2].find_all("td")
    if len(data_tds) < 9:
        return None

    keys = [
        "登録者数",
        "A",
        "B",
        "C",
        "D",
        "F",
        "他",
        "評点平均値",
        "備考",
    ]
    values = [td.get_text(strip=True) for td in data_tds[: len(keys)]]

    return dict(zip(keys, values))


def parse_syllabus_html(path: Path) -> dict:
    """
    単一 HTML ファイルから必要な情報を dict で返す。
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    base = parse_course_block(soup)

    # 概要
    overview = extract_section_text(soup, "＜概要")
    # 到達目標
    goals = extract_section_text(soup, "＜到達目標")

    # 成績評価基準
    eval_criteria = parse_evaluation_criteria(soup)

    # 成績評価結果
    grade_results = parse_grade_results(soup)

    record = {
        "科目名": base.get("科目名"),
        "開講学期": base.get("開講学期"),
        "曜日・時限": base.get("曜日・時限"),
        "単位数": base.get("単位数"),
        "授業形態": base.get("授業形態"),
        "実施方法": base.get("実施方法"),
        "教授名": base.get("教授名"),
        "到達目標": goals,
        "概要": overview,
        "成績評価基準": eval_criteria,
        "成績評価結果": grade_results,
    }
    return record


def main():
    all_records = []

    # このスクリプトと同じディレクトリ内の *.html を全部読む
    for path in SYLLABUS_DIR.glob("*.html"):
        try:
            rec = parse_syllabus_html(path)
            all_records.append(rec)
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    # --- JSON 出力 ---
    json_path = SYLLABUS_DIR / "syllabus_parsed.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # --- CSV 出力 ---
    csv_path = SYLLABUS_DIR / "syllabus_parsed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()

        for rec in all_records:
            row = {}
            for key in FIELD_ORDER:
                val = rec.get(key)
                # 成績評価基準 / 成績評価結果 は list/dict → JSON 文字列に変換
                if key in ("成績評価基準", "成績評価結果"):
                    if val is None:
                        row[key] = ""
                    else:
                        row[key] = json.dumps(val, ensure_ascii=False)
                else:
                    row[key] = "" if val is None else val
            writer.writerow(row)

    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
