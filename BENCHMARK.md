# AnyPhotoDoc6300 Instructions for Use

This document explains the file organization and naming rules of **AnyPhotoDoc6300 benchmark**, helping users quickly understand and use it.

---

## 1. Data structures and file naming rules

All pictures are named with a ** number sequence **, for example:

- `1_1_1_1_1.JPG`

The meanings are as follows:
- The first digit: indicates **Layout Category**, such as document types (Single-column Layout Paper,Double-column Layout Paper,Bound Book, etc.).
- The second number: represents ** typical warping patterns ** i.e.,crumples(1),curves(2),folds(3)
- The third number: indicates **Environment Lighting** (1, 2, and 3 are dim light, daylight, and indoor light respectively).
- The fourth digit: indicates the "instance number" (for different document samples under the same category).
- The fifth digit: indicates the ** version/enhancement number ** (different shooting angles).




---

## 2. The meaning of 'Init 1-8 '
1:Single-column Layout Paper
2:Complex Layout Paper 
3:Invoice
4:Education Script
5:Bound Book
6:Double-column Layout Paper
7:Magazine
8:Bill




---

## Table: Statistics of AnyPhotoDoc6300

**Flat Source**: flat source document obtained by scanner
**Photo Doc.**: Number of Photographic Document Images
**WP**: Warping Pattern
**Domains**: LC (Layout Category), EL (Environment Lighting), CA (Capture Angle)

| LC (Layout Category)       | Source Flat | WP | EL | CA | Photo Doc. |
| -------------------------- | ----------- | -- | -- | -- | ---------- |
| Single-column Layout Paper | 50          | ×3 | ×3 | ×2 | 900        |
| Double-column Layout Paper | 50          | ×3 | ×3 | ×2 | 900        |
| Bound Book                 | 100         | ×1 | ×3 | ×2 | 600        |
| Complex Layout Paper       | 50          | ×3 | ×3 | ×2 | 900        |
| Bill                       | 51          | ×3 | ×3 | ×2 | 306        |
| Invoice                    | 50          | ×3 | ×3 | ×2 | 900        |
| Magazine                   | 50          | ×3 | ×3 | ×2 | 900        |
| Education Script           | 50          | ×3 | ×3 | ×2 | 900        |
| **Total**                  |             |    |    |    | **6306**   |

---


📌 For more details, please refer to the original paper and the link in 'README.md'.
