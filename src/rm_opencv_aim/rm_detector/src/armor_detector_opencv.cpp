#include "rm_detector/armor_detector_opencv.hpp"

// 计算两个点之间的距离
double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return cv::norm(p1 - p2);
}

// 调整宽高和角度的函数
std::pair<cv::Size2f, double> adjust(const cv::Size2f& w_h, double angle) {
    float w = w_h.width;
    float h = w_h.height;

    if (w > h) {
        std::swap(w, h);
        if (angle >= 0) {
            angle -= 90;
        } else {
            angle += 90;
        }
    }

    return std::make_pair(cv::Size2f(w, h), angle);
}

// 将角度转换为斜率的函数
double angle_to_slope(double angle_degrees) {
    double angle_radians = angle_degrees * M_PI / 180.0;
    return std::tan(angle_radians);
}

// Light 类的构造函数实现
Light::Light(const cv::Point2f& up, const cv::Point2f& down, double angle, int color)
    : up(up), down(down), angle(angle), color(color) {
    cx = static_cast<int>(std::abs(up.x - down.x) / 2 + std::min(up.x, down.x));
    cy = static_cast<int>(std::abs(up.y - down.y) / 2 + std::min(up.y, down.y));
    height = calculate_distance(up, down);
}

Armor::Armor(const Light& light1, const Light& light2, NumberClassifier::Result res)
    : light1_up(light1.up), light1_down(light1.down),
      light2_up(light2.up), light2_down(light2.down),
      res(res) {

    color = light1.color;
    armor_id = get_id();
}

int Armor::get_id() const {
    if(color == 0){
        if(res.class_id == 0){
            return 6;
        }
        if(res.class_id == 1){
            return 8;
        }
        if(res.class_id == 2){
            return 11;
        }
    }
    else if(color == 1){
        if(res.class_id == 0){
            return 0;
        }
        if(res.class_id == 1){
            return 2;
        }
        if(res.class_id == 2){
            return 5;
        }
    }
    return -1;
}

// ArmorDetector 类的实现
ArmorDetector::ArmorDetector(int detect_color, int display_mode, int binary_val, const Params& params)
    : binary_val(binary_val), color(detect_color), display_mode(display_mode), params(params) {}

void ArmorDetector::update_light_area_min(int new_light_area_min) {
    params.light_area_min = new_light_area_min;
}

void ArmorDetector::update_light_h_w_ratio(double new_light_h_w_ratio) {
    params.light_h_w_ratio = new_light_h_w_ratio;
}

void ArmorDetector::update_light_angle_min(int new_light_angle_min) {
    params.light_angle_min = new_light_angle_min;
}

void ArmorDetector::update_light_angle_max(int new_light_angle_max) {
    params.light_angle_max = new_light_angle_max;
}

void ArmorDetector::update_light_red_ratio(float new_light_red_ratio) {
    params.light_red_ratio = new_light_red_ratio;
}

void ArmorDetector::update_light_blue_ratio(float new_light_blue_ratio) {
    params.light_blue_ratio = new_light_blue_ratio;
}

void ArmorDetector::update_height_rate_tol(float new_height_rate_tol) {
    params.height_rate_tol = new_height_rate_tol;
}

void ArmorDetector::update_height_multiplier_min(float new_height_multiplier_min) {
    params.height_multiplier_min = new_height_multiplier_min;
}

void ArmorDetector::update_height_multiplier_max(float new_height_multiplier_max) {
    params.height_multiplier_max = new_height_multiplier_max;
}

void ArmorDetector::update_binary_val(int new_binary_val) {
    binary_val = new_binary_val;
}

void ArmorDetector::update_detect_color(int new_color) {
    color = new_color;
}

void ArmorDetector::update_display_mode(int new_display_mode) {
    display_mode = new_display_mode;
}

cv::Mat ArmorDetector::process(const cv::Mat& img_input) {
    if (img_input.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return cv::Mat();
    }
    img = img_input.clone();
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, img_binary, binary_val, 255, cv::THRESH_BINARY);
    return img_binary;
}

std::vector<Light> ArmorDetector::find_lights(const cv::Mat& img_binary_input) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::RotatedRect> is_lights;
    std::vector<Light> lights_found;

    cv::findContours(img_binary_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= params.light_area_min) {
            cv::RotatedRect min_rect = cv::minAreaRect(contour);
            cv::Size2f w_h = min_rect.size;
            double angle = min_rect.angle;
            std::tie(w_h, angle) = adjust(w_h, angle);
            if (w_h.height / w_h.width < params.light_h_w_ratio) {
                continue;
            }
            if (angle >= params.light_angle_min && angle <= params.light_angle_max) {
                cv::RotatedRect rect(min_rect.center, w_h, static_cast<float>(angle));
                is_lights.push_back(rect);
            }
        }
    }

    for (const auto& rect : is_lights) {
        cv::Point2f box[4];
        rect.points(box);

        int up_x = static_cast<int>(std::abs(box[0].x - box[3].x) / 2 + std::min(box[0].x, box[3].x));
        int up_y = static_cast<int>(std::abs(box[0].y - box[3].y) / 2 + std::min(box[0].y, box[3].y));
        cv::Point2f up(up_x, up_y);

        int down_x = static_cast<int>(std::abs(box[1].x - box[2].x) / 2 + std::min(box[1].x, box[2].x));
        int down_y = static_cast<int>(std::abs(box[1].y - box[2].y) / 2 + std::min(box[1].y, box[2].y));
        cv::Point2f down(down_x, down_y);

        int length = static_cast<int>(calculate_distance(up, down));
        
        if (length <= 0) continue;  // 避免空 ROI
        
        cv::Mat roi(1, length, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int i = 0; i < length; ++i) {
            float t = static_cast<float>(i) / length;
            int current_x = static_cast<int>(up_x + (down_x - up_x) * t);
            int current_y = static_cast<int>(up_y + (down_y - up_y) * t);

            if (current_x >= 0 && current_x < img.cols && current_y >= 0 && current_y < img.rows) {
                roi.at<cv::Vec3b>(0, i) = img.at<cv::Vec3b>(current_y, current_x);
            }
        }

        int sum_b = cv::sum(roi)[0];
        int sum_r = cv::sum(roi)[2];

        if ((color == 1 || color == 2) && sum_b > sum_r * params.light_blue_ratio) {
            lights_found.push_back(Light(up, down, rect.angle, 1));
        } else if ((color == 0 || color == 2) && sum_r > sum_b * params.light_red_ratio) {
            lights_found.push_back(Light(up, down, rect.angle, 0));
        }
    }

    lights = lights_found;
    return lights;
}

std::pair<cv::Point2f, cv::Point2f> ArmorDetector::stretch_point(const cv::Point2f& up, const cv::Point2f& down, float ratio) const {
    cv::Point2f center = (up + down) * 0.5f;
    cv::Point2f v = down - up;
    float h = cv::norm(v);
    if (h < 1e-3) {
        return {up, down};  // 避免除零
    }

    float scale = (1.0f / ratio) * 0.5f;  // 上下各拉一半
    cv::Point2f v_scaled = v * scale;

    return {center - v_scaled, center + v_scaled};
}

NumberClassifier::Result ArmorDetector::get_armor_result(const Light& light1, const Light& light2) {
    NumberClassifier::Result res;

    auto l1 = stretch_point(light1.up, light1.down, 0.45f);
    auto l2 = stretch_point(light2.up, light2.down, 0.45f);

    cv::Point2f src_armor_pts[4] = {
        l1.second, // left-down
        l1.first,  // left-up
        l2.first,  // right-up
        l2.second  // right-down
    };

    cv::Mat H = cv::getPerspectiveTransform(src_armor_pts, dst_armor_pts);
    cv::Mat roi;
    cv::warpPerspective(img, roi, H, cv::Size(134, int(57 / 0.45f)));
    img_armor = roi;
    if (classifier_ && !roi.empty()) {
        res = classifier_->classify(roi);
    }

    return res; // 只返回结果（id + 置信度）
}

NumberClassifier::Result ArmorDetector::is_close(const Light& light1, const Light& light2) {
    NumberClassifier::Result res;
    // 计算公共变量
    float height = std::max(light1.height, light2.height);
    float height_rate = height / std::min(light1.height, light2.height);
    // 如果高度比例不符合，直接返回
    if (height_rate >= params.height_rate_tol) {
        return res;
    }
    double distance = calculate_distance({static_cast<float>(light1.cx), static_cast<float>(light1.cy)}, {static_cast<float>(light2.cx), static_cast<float>(light2.cy)});

    if ((distance > height * params.height_multiplier_min && distance < height * params.height_multiplier_max)
         || (distance > height * 1.56f * params.height_multiplier_min && distance < height * 1.76f * params.height_multiplier_max)
        ) {
            res = get_armor_result(light1, light2);
    }
    return res;
}

std::vector<Armor> ArmorDetector::is_armor(const std::vector<Light>& lights) {
    std::vector<Armor> armors_found;

    if (lights.size() < 2) {
        return armors_found;
    }

    // 复制一份 lights，并按中心 x 排序（左 → 右）
    std::vector<Light> sorted_lights = lights;
    std::sort(sorted_lights.begin(), sorted_lights.end(),
        [](const Light& a, const Light& b) {
            return a.cx < b.cx;
        });

    // 相邻配对
    for (size_t i = 0; i + 1 < sorted_lights.size(); i++) {
        const Light& left  = sorted_lights[i];
        const Light& right = sorted_lights[i + 1];

        if (left.color == right.color) {
            NumberClassifier::Result res = is_close(left, right);
            if (res.class_id >= 0 && res.class_id < 3) {
                armors_found.push_back(Armor(left, right, res));
            }
        }
    }

    armors = armors_found;
    return armors;
}

cv::Mat ArmorDetector::draw_lights(cv::Mat img_draw) {
    for (const auto& light : lights) {
        if (light.color == 0) {
            cv::line(img_draw, light.up, light.down, cv::Scalar(0, 100, 255), 1);
            cv::circle(img_draw, cv::Point(light.cx, light.cy), 1, cv::Scalar(255, 0, 0), -1);
        } else if (light.color == 1) {
            cv::line(img_draw, light.up, light.down, cv::Scalar(200, 71, 90), 1);
            cv::circle(img_draw, cv::Point(light.cx, light.cy), 1, cv::Scalar(0, 0, 255), -1);
        }
    }
    return img_draw;
}

cv::Mat ArmorDetector::draw_armors(cv::Mat img_draw) {
    for (const auto& armor : armors) {
        // 绘制交叉线
        if (armor.color == 0) {
            cv::line(img_draw, armor.light1_up, armor.light2_down, cv::Scalar(128, 0, 128), 1);
            cv::line(img_draw, armor.light2_up, armor.light1_down, cv::Scalar(128, 0, 128), 1);
        } else if (armor.color == 1) {
            cv::line(img_draw, armor.light1_up, armor.light2_down, cv::Scalar(255, 255, 0), 1);
            cv::line(img_draw, armor.light2_up, armor.light1_down, cv::Scalar(255, 255, 0), 1);
        }

        // 设置文本参数
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Scalar textColor(255, 0, 255); // 粉色 BGR

        // 文本位置
        cv::Point textOrg = armor.light1_up; 
        int lineHeight = 15; // 每行间距，可以根据需要调整

        // 绘制三行文本
        cv::putText(img_draw, std::to_string(armor.armor_id), textOrg, fontFace, fontScale, textColor, thickness);
        cv::putText(img_draw, std::to_string(armor.res.class_id), cv::Point(textOrg.x, textOrg.y + lineHeight),
                    fontFace, fontScale, textColor, thickness);
        cv::putText(img_draw, std::to_string(armor.res.confidence), cv::Point(textOrg.x, textOrg.y + 2 * lineHeight),
                    fontFace, fontScale, textColor, thickness);
    }
    return img_draw;
}

cv::Mat ArmorDetector::draw_img() {
    cv::Mat img_draw = img.clone();
    img_draw = draw_armors(img_draw);
    img_drawn = draw_lights(img_draw);
    return img_drawn;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> ArmorDetector::display() {
    if (display_mode == 1) {
        return std::make_tuple(img_binary, cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)));
    } else if (display_mode == 2) {
        img_drawn = draw_img();
        return std::make_tuple(img_binary, img_drawn, img_armor);
    } else if (display_mode == 0) {
        return std::make_tuple(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)));
    } else {
        std::cerr << "Invalid display mode" << std::endl;
        return std::make_tuple(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)));
    }
}

std::vector<Armor> ArmorDetector::detect_armors(const cv::Mat& img_input) {
    img_binary = process(img_input);
    lights = find_lights(img_binary);
    armors = is_armor(lights);
    return armors;
}