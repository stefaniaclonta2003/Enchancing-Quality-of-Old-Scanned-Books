#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <queue>
#include <sstream>

using namespace cv;
using namespace std;

struct TextBlockClearText {
    Rect boundingBox;
    string transcribedText;
    bool isValidated;

    TextBlockClearText(Rect box) : boundingBox(box), isValidated(false) {}
};

static vector<TextBlockClearText>* g_textBlocksClearText = nullptr;
static int g_selectedBlockClearText = -1;
static double g_displayScaleClearText = 1.0;
static bool g_needsUpdateClearText = true;

Mat resizeForDisplayClearText(const Mat& img, int maxWidth = 1000, int maxHeight = 700) {
    if (img.empty()) return img;

    Mat resized;
    double scale = 1.0;

    if (img.cols > maxWidth || img.rows > maxHeight) {
        double scaleX = (double)maxWidth / img.cols;
        double scaleY = (double)maxHeight / img.rows;

        if (scaleX < scaleY)
            scale = scaleX;
        if (scaleX > scaleY)
            scale = scaleY;

        int newWidth = (int)(img.cols * scale);
        int newHeight = (int)(img.rows * scale);

        resize(img, resized, Size(newWidth, newHeight), 0, 0, INTER_AREA);
        return resized;
    }

    return img.clone();
}

Mat grayscale_to_BW_autoClearText(Mat src) {
    Mat dst(src.rows, src.cols, CV_8UC1);

    int hist[256] = { 0 };
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            hist[src.at<uchar>(i, j)]++;
        }
    }

    int T = 128;
    int T_prev = 0;
    float error = 0.1f;

    while (abs(T - T_prev) >= error) {
        T_prev = T;
        int count1 = 0, count2 = 0;
        int sum1 = 0, sum2 = 0;

        for (int i = 0; i < 256; i++) {
            if (i <= T) {
                count1 += hist[i];
                sum1 += i * hist[i];
            }
            else {
                count2 += hist[i];
                sum2 += i * hist[i];
            }
        }

        float mean1;
        if (count1 > 0) {
            mean1 = (float)sum1 / count1;
        }
        else {
            mean1 = 0;
        }

        float mean2;
        if (count2 > 0) {
            mean2 = (float)sum2 / count2;
        }
        else {
            mean2 = 0;
        }

        T = (int)((mean1 + mean2) / 2);
    }

    printf("Prag calculat automat: %d\n", T);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i, j) < T) {
                dst.at<uchar>(i, j) = 0;
            }
            else {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    return dst;
}

Mat dilateCustomClearText(Mat src, int structSize = 5) {
    Mat dst = src.clone();
    int half = structSize / 2;

    for (int i = half; i < src.rows - half; i++) {
        for (int j = half; j < src.cols - half; j++) {
            if (src.at<uchar>(i, j) == 0) {
                for (int di = -half; di <= half; di++) {
                    for (int dj = -half; dj <= half; dj++) {
                        int ni = i + di;
                        int nj = j + dj;

                        if (ni >= 0 && ni < dst.rows && nj >= 0 && nj < dst.cols) {
                            dst.at<uchar>(ni, nj) = 0;
                        }
                    }
                }
            }
        }
    }

    return dst;
}

int labelConnectedComponentsClearText(Mat img, Mat& labels, vector<Rect>& boundingBoxes) {
    int height = img.rows;
    int width = img.cols;
    int nr = 0;

    labels = Mat::zeros(height, width, CV_32SC1);
    boundingBoxes.clear();

    int di[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    int dj[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
                nr++;
                queue<Point> Q;
                labels.at<int>(i, j) = nr;
                Q.push(Point(j, i));

                int minX = j, maxX = j, minY = i, maxY = i;

                while (!Q.empty()) {
                    Point q = Q.front();
                    Q.pop();
                    int qi = q.y;
                    int qj = q.x;

                    if (qj < minX) minX = qj;
                    if (qj > maxX) maxX = qj;
                    if (qi < minY) minY = qi;
                    if (qi > maxY) maxY = qi;

                    for (int k = 0; k < 8; k++) {
                        int ni = qi + di[k];
                        int nj = qj + dj[k];

                        if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                            if (img.at<uchar>(ni, nj) == 0 && labels.at<int>(ni, nj) == 0) {
                                labels.at<int>(ni, nj) = nr;
                                Q.push(Point(nj, ni));
                            }
                        }
                    }
                }

                Rect bbox(minX, minY, maxX - minX + 1, maxY - minY + 1);
                boundingBoxes.push_back(bbox);
            }
        }
    }

    return nr;
}

Mat simpleInpaintingClearText(Mat original, Mat mask) {
    Mat result = original.clone();

    for (int iter = 0; iter < 10; iter++) {
        Mat temp = result.clone();

        for (int i = 1; i < mask.rows - 1; i++) {
            for (int j = 1; j < mask.cols - 1; j++) {
                if (mask.at<uchar>(i, j) == 0) {
                    int count = 0;
                    Vec3f sum(0, 0, 0);

                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            if (di == 0 && dj == 0) continue;

                            int ni = i + di;
                            int nj = j + dj;

                            if (mask.at<uchar>(ni, nj) == 255) {
                                Vec3b pixel = result.at<Vec3b>(ni, nj);
                                sum[0] += pixel[0];
                                sum[1] += pixel[1];
                                sum[2] += pixel[2];
                                count++;
                            }
                        }
                    }

                    if (count > 0) {
                        temp.at<Vec3b>(i, j) = Vec3b(
                            (uchar)(sum[0] / count),
                            (uchar)(sum[1] / count),
                            (uchar)(sum[2] / count)
                        );
                    }
                }
            }
        }
        result = temp;
    }

    return result;
}

void onMouseCallbackClearText(int event, int x, int y, int flags, void* userdata) {

    if (event == EVENT_LBUTTONDOWN && g_textBlocksClearText != nullptr) {

        Point originalPoint((int)(x / g_displayScaleClearText), (int)(y / g_displayScaleClearText));

        int oldSelected = g_selectedBlockClearText;
        g_selectedBlockClearText = -1;

        for (size_t i = 0; i < g_textBlocksClearText->size(); i++) {
            Rect block = (*g_textBlocksClearText)[i].boundingBox;

            if (block.contains(originalPoint)) {
                g_selectedBlockClearText = (int)i;
                break;
            }
        }

        if (oldSelected != g_selectedBlockClearText) {
            g_needsUpdateClearText = true;
        }
    }

    if (event == EVENT_RBUTTONDOWN) {
        printf("Click dreapta - deselect\n");
        if (g_selectedBlockClearText != -1) {
            g_selectedBlockClearText = -1;
            g_needsUpdateClearText = true;
        }
    }
}

void testClearTextWithMouseSelection() {

    char fname[MAX_PATH];
    if (!openFileDlg(fname)) {
        printf("Nu a fost selectata nicio imagine.\n");
        return;
    }

    Mat originalImage = imread(fname, IMREAD_COLOR);
    if (originalImage.empty()) {
        printf("Nu am putut incarca imaginea: %s\n", fname);
        return;
    }

    printf("\n=== DETECTIA BLOCURILOR DE TEXT ===\n");

    Mat grayImage(originalImage.rows, originalImage.cols, CV_8UC1);
    for (int i = 0; i < originalImage.rows; i++) {
        for (int j = 0; j < originalImage.cols; j++) {
            Vec3b pixel = originalImage.at<Vec3b>(i, j);
            uchar gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
            grayImage.at<uchar>(i, j) = gray;
        }
    }

    Mat binaryImage = grayscale_to_BW_autoClearText(grayImage);

    Mat labels;
    vector<Rect> boundingBoxes;
    int numComponents = labelConnectedComponentsClearText(binaryImage, labels, boundingBoxes);

    vector<bool> isTextComponent(numComponents + 1, false);
    for (int i = 0; i < numComponents; i++) {
        Rect box = boundingBoxes[i];
        int area = box.area();

        if (area > 10 && area < 5000 &&
            box.width < originalImage.cols / 4 &&
            box.height < originalImage.rows / 10 &&
            box.width > 5 && box.height > 5) {
            isTextComponent[i + 1] = true;
        }
    }

    Mat dilatedImage = Mat::zeros(binaryImage.size(), CV_8UC1);
    dilatedImage.setTo(255);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label > 0 && isTextComponent[label]) {
                dilatedImage.at<uchar>(i, j) = 0;
            }
        }
    }

    dilatedImage = dilateCustomClearText(dilatedImage, 5);

    Mat horizontalDilated = dilatedImage.clone();
    for (int i = 0; i < dilatedImage.rows; i++) {
        for (int j = 0; j < dilatedImage.cols; j++) {
            if (dilatedImage.at<uchar>(i, j) == 0) {
                for (int extend = 1; extend <= 10; extend++) {
                    if (j + extend < horizontalDilated.cols) {
                        horizontalDilated.at<uchar>(i, j + extend) = 0;
                    }
                    if (j - extend >= 0) {
                        horizontalDilated.at<uchar>(i, j - extend) = 0;
                    }
                }
            }
        }
    }
    dilatedImage = horizontalDilated;

    vector<Rect> finalBoundingBoxes;
    vector<TextBlockClearText> textBlocks;
    int finalComponents = labelConnectedComponentsClearText(dilatedImage, labels, finalBoundingBoxes);

    for (int i = 0; i < finalComponents; i++) {
        Rect box = finalBoundingBoxes[i];
        if (box.area() > 200 && box.area() < originalImage.rows * originalImage.cols / 8) {
            if (box.width > 20 && box.height > 10) {
                textBlocks.push_back(TextBlockClearText(box));
            }
        }
    }

    printf("REZULTAT: %zu blocuri de text detectate\n", textBlocks.size());

    imshow("1. Original", resizeForDisplayClearText(originalImage));
    imshow("2. Grayscale", resizeForDisplayClearText(grayImage));
    imshow("3. Binary", resizeForDisplayClearText(binaryImage));
    imshow("4. Dilated", resizeForDisplayClearText(dilatedImage));

    Mat blocksDisplay = originalImage.clone();
    for (size_t i = 0; i < textBlocks.size(); i++) {
        rectangle(blocksDisplay, textBlocks[i].boundingBox, Scalar(0, 0, 255), 2);
        putText(blocksDisplay, to_string(i + 1),
            Point(textBlocks[i].boundingBox.x, textBlocks[i].boundingBox.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
    }
    imshow("5. Blocuri Detectate", resizeForDisplayClearText(blocksDisplay));

    printf("Apasa orice tasta pentru transcrierea cu mouse...\n");
    waitKey();
    destroyAllWindows();

    printf("\n=== TRANSCRIEREA CU SELECTIE MOUSE ===\n");
    printf("Instructiuni:\n");
    printf("- Click STANGA pe un bloc pentru a-l selecta\n");
    printf("- Click DREAPTA pentru a deselecta\n");
    printf("- Apasa 't' pentru a transcrie blocul selectat\n");
    printf("- Apasa 'd' pentru a sterge blocul selectat\n");
    printf("- Apasa 's' pentru a salva si continua\n");

    g_textBlocksClearText = &textBlocks;
    g_selectedBlockClearText = -1;
    g_needsUpdateClearText = true;

    Mat originalDisplay = resizeForDisplayClearText(originalImage);
    g_displayScaleClearText = (double)originalDisplay.cols / originalImage.cols;

    namedWindow("ClearText - Transcriere cu Mouse", WINDOW_AUTOSIZE);
    setMouseCallback("ClearText - Transcriere cu Mouse", onMouseCallbackClearText, nullptr);

    while (true) {
        if (g_needsUpdateClearText) {
            Mat display = originalDisplay.clone();

            for (size_t i = 0; i < textBlocks.size(); i++) {
                Scalar color;
                int thickness = max(2, (int)(2 * g_displayScaleClearText));

                if (textBlocks[i].isValidated) {
                    color = Scalar(0, 255, 0);
                }
                else {
                    color = Scalar(0, 0, 255);
                }

                if ((int)i == g_selectedBlockClearText) {
                    color = Scalar(255, 255, 0);
                    thickness = max(4, (int)(4 * g_displayScaleClearText));
                }

                Rect displayRect((int)(textBlocks[i].boundingBox.x * g_displayScaleClearText),
                    (int)(textBlocks[i].boundingBox.y * g_displayScaleClearText),
                    (int)(textBlocks[i].boundingBox.width * g_displayScaleClearText),
                    (int)(textBlocks[i].boundingBox.height * g_displayScaleClearText));

                rectangle(display, displayRect, color, thickness);
                putText(display, to_string(i + 1),
                    Point(displayRect.x, displayRect.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.6 * g_displayScaleClearText, color, max(1, (int)(2 * g_displayScaleClearText)));

                if (textBlocks[i].isValidated && !textBlocks[i].transcribedText.empty()) {
                    string preview = textBlocks[i].transcribedText.substr(0, 15);
                    if (textBlocks[i].transcribedText.length() > 15) preview += "...";
                    putText(display, preview,
                        Point(displayRect.x, displayRect.y + displayRect.height + (int)(15 * g_displayScaleClearText)),
                        FONT_HERSHEY_SIMPLEX, 0.35 * g_displayScaleClearText, Scalar(255, 255, 255), max(1, (int)(1 * g_displayScaleClearText)));
                }
            }

            putText(display, "Blocuri: " + to_string(textBlocks.size()),
                Point(10, (int)(30 * g_displayScaleClearText)), FONT_HERSHEY_SIMPLEX, 0.7 * g_displayScaleClearText, Scalar(255, 255, 0), max(1, (int)(2 * g_displayScaleClearText)));

            int transcribed = 0;
            for (const auto& block : textBlocks) {
                if (block.isValidated) transcribed++;
            }
            putText(display, "Transcrise: " + to_string(transcribed),
                Point(10, (int)(60 * g_displayScaleClearText)), FONT_HERSHEY_SIMPLEX, 0.7 * g_displayScaleClearText, Scalar(0, 255, 0), max(1, (int)(2 * g_displayScaleClearText)));

            if (g_selectedBlockClearText >= 0 && g_selectedBlockClearText < (int)textBlocks.size()) {
                putText(display, "Selectat: Bloc " + to_string(g_selectedBlockClearText + 1),
                    Point(10, (int)(90 * g_displayScaleClearText)), FONT_HERSHEY_SIMPLEX, 0.7 * g_displayScaleClearText, Scalar(255, 255, 0), max(1, (int)(2 * g_displayScaleClearText)));
            }
            else {
                putText(display, "Selectat: niciun bloc",
                    Point(10, (int)(90 * g_displayScaleClearText)), FONT_HERSHEY_SIMPLEX, 0.7 * g_displayScaleClearText, Scalar(128, 128, 128), max(1, (int)(2 * g_displayScaleClearText)));
            }

            putText(display, "Click pe bloc=selecteaza, t=transcrie, d=sterge, s=salveaza",
                Point(10, (int)(120 * g_displayScaleClearText)), FONT_HERSHEY_SIMPLEX, 0.4 * g_displayScaleClearText, Scalar(150, 150, 255), max(1, (int)(1 * g_displayScaleClearText)));

            imshow("ClearText - Transcriere cu Mouse", display);
            g_needsUpdateClearText = false;
        }

        char key = waitKey(30) & 0xFF;
        if (key == 27) {
            printf("ESC apasat - iesire din validare.\n");
            break;
        }

        if (key == 't' || key == 'T') {
            if (g_selectedBlockClearText >= 0 && g_selectedBlockClearText < (int)textBlocks.size()) {
                printf("\n=== TRANSCRIEREA BLOCULUI %d ===\n", g_selectedBlockClearText + 1);
                printf("Introdu textul din aceasta regiune: ");

                string text;
                getline(cin, text);

                if (!text.empty()) {
                    textBlocks[g_selectedBlockClearText].transcribedText = text;
                    textBlocks[g_selectedBlockClearText].isValidated = true;
                    printf("Text salvat: \"%s\"\n", text.c_str());
                    g_needsUpdateClearText = true;
                }
                else {
                    printf("Text gol - transcrierea anulata.\n");
                }
            }
            else {
                printf("Selecteaza un bloc mai intai cu mouse-ul!\n");
            }
        }

        if (key == 'd' || key == 'D') {
            if (g_selectedBlockClearText >= 0 && g_selectedBlockClearText < (int)textBlocks.size()) {
                printf("Sterg blocul %d\n", g_selectedBlockClearText + 1);
                textBlocks.erase(textBlocks.begin() + g_selectedBlockClearText);
                g_selectedBlockClearText = -1;
                g_needsUpdateClearText = true;
                printf("Bloc sters cu succes.\n");
            }
            else {
                printf("Selecteaza un bloc pentru stergere!\n");
            }
        }

        if (key == 's' || key == 'S') {
            printf("Salvez si continui...\n");
            break;
        }
    }

    g_textBlocksClearText = nullptr;
    g_selectedBlockClearText = -1;
    destroyWindow("ClearText - Transcriere cu Mouse");

    printf("\n=== RECONSTRUIREA FUNDALULUI ===\n");

    Mat mask = Mat::zeros(originalImage.size(), CV_8UC1);

    for (const auto& block : textBlocks) {
        Mat blockRegion = binaryImage(block.boundingBox);
        Mat maskRegion = mask(block.boundingBox);

        for (int i = 0; i < blockRegion.rows; i++) {
            for (int j = 0; j < blockRegion.cols; j++) {
                if (blockRegion.at<uchar>(i, j) == 0) {
                    maskRegion.at<uchar>(i, j) = 0;
                }
                else {
                    maskRegion.at<uchar>(i, j) = 255;
                }
            }
        }
    }

    Mat backgroundOnly = simpleInpaintingClearText(originalImage, mask);
    Mat result = backgroundOnly.clone();

    for (size_t i = 0; i < textBlocks.size(); i++) {
        if (!textBlocks[i].isValidated || textBlocks[i].transcribedText.empty()) {
            continue;
        }

        const TextBlockClearText& block = textBlocks[i];
        printf("Bloc %zu: rendering \"%s\"\n", i + 1,
            block.transcribedText.substr(0, 30).c_str());

        vector<double> fontSizes = { 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0 };
        double bestFontScale = 0.8;
        int thickness = 1.2;

        int blockWidth = block.boundingBox.width;
        int blockHeight = block.boundingBox.height;

        printf("   Dimensiuni bloc: %dx%d pixeli\n", blockWidth, blockHeight);

        vector<string> words;
        istringstream iss(block.transcribedText);
        string word;
        while (iss >> word) {
            words.push_back(word);
        }

        for (double testFont : fontSizes) {
            vector<string> testLines;
            string currentLine = "";
            bool fontFits = true;

            for (const string& w : words) {
                string testLine;
                if (currentLine.empty()) {
                    testLine = w;
                }
                else {
                    testLine = currentLine + " " + w;
                }

                Size testSize = getTextSize(testLine, FONT_HERSHEY_SIMPLEX,
                    testFont, thickness, nullptr);

                if (testSize.width <= blockWidth - 6) {
                    currentLine = testLine;
                }
                else {
                    if (!currentLine.empty()) {
                        testLines.push_back(currentLine);
                        currentLine = w;
                    }
                    else {
                        fontFits = false;
                        break;
                    }
                }
            }
            if (!currentLine.empty()) {
                testLines.push_back(currentLine);
            }

            if (fontFits && !testLines.empty()) {
                Size sampleSize = getTextSize("Ag", FONT_HERSHEY_SIMPLEX, testFont, thickness, nullptr);
                int lineHeight = sampleSize.height + 2;
                int totalHeight = testLines.size() * lineHeight;

                if (totalHeight <= blockHeight - 6) {
                    bestFontScale = testFont;
                    printf("   Font %0.1f se potriveste: %d linii, inaltime %d\n", testFont, (int)testLines.size(), totalHeight);
                }
                else {
                    printf("   Font %0.1f prea inalt: %d linii, inaltime %d > %d\n", testFont, (int)testLines.size(), totalHeight, blockHeight - 6);
                    break;
                }
            }
            else {
                printf("   Font %0.1f prea lat\n", testFont);
                break;
            }
        }

        if (bestFontScale < 0.8) {
            bestFontScale = 0.8;
            printf("   FORTAT: Font marit la %.1f pentru vizibilitate\n", bestFontScale);
        }

        printf("   Font final: %.1f, thickness: %d\n", bestFontScale, thickness);

        vector<string> finalLines;
        string currentLine = "";

        for (const string& w : words) {
            string testLine;
            if (currentLine.empty()) {
                testLine = w;
            }
            else {
                testLine = currentLine + " " + w;
            }
            Size testSize = getTextSize(testLine, FONT_HERSHEY_SIMPLEX,
                bestFontScale, thickness, nullptr);

            if (testSize.width <= blockWidth - 6) {
                currentLine = testLine;
            }
            else {
                if (!currentLine.empty()) {
                    finalLines.push_back(currentLine);
                    currentLine = w;
                }
                else {
                    finalLines.push_back(w);
                }
            }
        }
        if (!currentLine.empty()) {
            finalLines.push_back(currentLine);
        }

        Size sampleSize = getTextSize("Ag", FONT_HERSHEY_SIMPLEX, bestFontScale, thickness, nullptr);
        int lineHeight = sampleSize.height + 3;
        int startY = block.boundingBox.y + lineHeight;

        printf("   Randare: %d linii cu font %.1f\n", (int)finalLines.size(), bestFontScale);

        for (size_t l = 0; l < finalLines.size(); l++) {
            int y = startY + (int)l * lineHeight;

            if (y > originalImage.rows - 10) {
                printf("   Linia %d iese din imagine\n", (int)l);
                break;
            }

            Size lineSize = getTextSize(finalLines[l], FONT_HERSHEY_SIMPLEX,
                bestFontScale, thickness, nullptr);

            int padding = 3;
            Rect textBg(max(0, block.boundingBox.x - padding),
                max(0, y - lineSize.height - padding),
                min(originalImage.cols - block.boundingBox.x, lineSize.width + 2 * padding),
                lineSize.height + 2 * padding);

            rectangle(result, textBg, Scalar(255, 255, 255), -1);

            putText(result, finalLines[l],
                Point(block.boundingBox.x + 2, y),
                FONT_HERSHEY_SIMPLEX, bestFontScale, Scalar(0, 0, 0), thickness + 1);

            printf("   Linia %d: \"%s\" cu font %.1f\n", (int)l, finalLines[l].c_str(), bestFontScale);
        }

        printf("   FINALIZAT cu font %.1f!\n", bestFontScale);
    }

    printf("RENDERING COMPLET!\n");

    imshow("6. Fundal Reconstituit", resizeForDisplayClearText(backgroundOnly));
    imshow("7. Rezultat Final ClearText", resizeForDisplayClearText(result));
    moveWindow("6. Fundal Reconstituit", 50, 50);
    moveWindow("7. Rezultat Final ClearText", 400, 50);
    printf("              REZULTATE                 \n");
    printf("\n");
    printf("Blocuri detectate: %zu\n", textBlocks.size());

    int transcribed = 0;
    for (const auto& block : textBlocks) {
        if (block.isValidated) transcribed++;
    }
    printf("Blocuri transcrise: %d\n", transcribed);

    printf("\nTexte transcrise:\n");
    for (size_t i = 0; i < textBlocks.size(); i++) {
        if (textBlocks[i].isValidated) {
            printf("Bloc %zu: \"%s\"\n", i + 1, textBlocks[i].transcribedText.c_str());
        }
    }

    printf("\nApasa orice tasta pentru a inchide...\n");
    waitKey();
    destroyAllWindows();
}

int main() {
    int op;
    do {
        system("cls");
        destroyAllWindows();
        printf("\n");
        printf("                CLEARTEXT          \n");
        printf("\n");
        printf("Foloseste functiile implementate in lab:\n");
        printf("- Binarizare cu prag automat\n");
        printf("- Dilatare custom\n");
        printf("- Etichetare componente conexe\n");
        printf("- Inpainting simplu\n");
        printf("- SELECTIE CU MOUSE-UL pentru blocuri!\n");
        printf("\n");
        printf("Functionalitati mouse:\n");
        printf("+ Click stanga pe bloc = selectare\n");
        printf("+ Click dreapta = deselect\n");
        printf("+ Evidentiire vizuala a selectiei\n");
        printf("\n");
        printf("1 - ClearText cu selectie mouse\n");
        printf("0 - Exit\n");
        printf("\n");
        printf("Optiune: ");
        scanf("%d", &op);

        switch (op) {
        case 1:
            testClearTextWithMouseSelection();
            break;
        case 0:
            printf("La revedere!\n");
            break;
        default:
            printf("Optiune invalida!\n");
            break;
        }

        if (op != 0) {
            printf("\nApasa ENTER pentru a reveni la meniu...\n");
            char key;
            scanf("%c", &key);
        }
    } while (op != 0);

    return 0;
}