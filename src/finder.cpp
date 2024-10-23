#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <vector>
#include <cstdlib>
#include <rtabmap_msgs/srv/reset_pose.hpp>

// Define a hashing function for std::pair<int, int> for unordered_map
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};

// A* Planner class
class AStarPlanner { 
public:
    AStarPlanner(const cv::Mat& global_map, const int safe_distance) : global_map_(global_map), safe_distance_(safe_distance) {
        rows_ = global_map.rows;
        cols_ = global_map.cols;
        // Safe distance from obstacles (safe = 70cm == 14cells[5cm each])
        distances_to_obstacles_ = computeDistancesToObstacles(global_map_);
    }

    // A* planning method
    std::vector<std::pair<int, int>> plan(const std::pair<int, int>& start, const std::pair<int, int>& goal) {
        std::priority_queue<Node, std::vector<Node>, NodeComparator> open_set;
        std::unordered_map<std::pair<int, int>, double, PairHash> g_costs;
        std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash> came_from;

        // Push the start node with a cost of 0
        open_set.emplace(0, start);
        g_costs[start] = 0;

        while (!open_set.empty()) {
            auto current = open_set.top().cell;
            open_set.pop();

            // If the goal is reached, reconstruct the path
            if (current == goal) {
                return reconstructPath(came_from, current);
            }

            // Check the neighbors (up, down, left, right)
            for (const auto& [dx, dy] : neighbors) {
                std::pair<int, int> neighbor = {current.first + dx, current.second + dy};

                if (isValid(neighbor) && global_map_.at<uint8_t>(neighbor.first, neighbor.second) == 0) {
                    double new_cost = g_costs[current] + 1;

                    // Add cost if too close to obstacles
                    int dist_to_obstacle = distances_to_obstacles_[neighbor.first][neighbor.second];
                    if (dist_to_obstacle < safe_distance_) {
                        new_cost += (safe_distance_ - dist_to_obstacle);
                    }

                    if (!g_costs.count(neighbor) || new_cost < g_costs[neighbor]) {
                        g_costs[neighbor] = new_cost;
                        double priority = new_cost + heuristic(neighbor, goal);
                        open_set.emplace(priority, neighbor);
                        came_from[neighbor] = current;
                    }
                }
            }
        }
        return {}; // Return an empty path if there's no valid path
    }

private:
    struct Node {
        double cost;
        std::pair<int, int> cell;
        Node(double c, std::pair<int, int> p) : cost(c), cell(p) {}
    };

    struct NodeComparator {
        bool operator()(const Node& a, const Node& b) {
            return a.cost > b.cost; // Min-heap behavior
        }
    };

    const std::vector<std::pair<int, int>> neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows_, cols_, safe_distance_;
    cv::Mat global_map_;
    std::vector<std::vector<int>> distances_to_obstacles_;

    // Heuristic function for A*
    double heuristic(const std::pair<int, int>& cell, const std::pair<int, int>& goal) {
        return std::abs(cell.first - goal.first) + std::abs(cell.second - goal.second);
    }

    // Reconstruct the path by tracing back from the goal
    std::vector<std::pair<int, int>> reconstructPath(const std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash>& came_from,
                                                     std::pair<int, int> current) {
        std::vector<std::pair<int, int>> path;
        while (came_from.count(current)) {
            path.push_back(current);
            current = came_from.at(current);
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // Check if a cell is within bounds
    bool isValid(const std::pair<int, int>& cell) const {
        return cell.first >= 0 && cell.first < rows_ && cell.second >= 0 && cell.second < cols_;
    }

    // Compute distances to obstacles for each cell
    std::vector<std::vector<int>> computeDistancesToObstacles(const cv::Mat& grid) {
        std::vector<std::vector<int>> distances(rows_, std::vector<int>(cols_, std::numeric_limits<int>::max()));
        std::queue<std::pair<int, int>> queue;

        // Initialize distances for obstacle cells and add them to the queue
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (grid.at<uint8_t>(i, j) != 0) {
                    distances[i][j] = 0;
                    queue.emplace(i, j);
                }
            }
        }

        // Perform BFS to calculate distances from obstacles
        while (!queue.empty()) {
            auto [x, y] = queue.front();
            queue.pop();
            for (const auto& [dx, dy] : neighbors) {
                int nx = x + dx, ny = y + dy;
                if (isValid({nx, ny}) && distances[nx][ny] > distances[x][y] + 1) {
                    distances[nx][ny] = distances[x][y] + 1;
                    queue.emplace(nx, ny);
                }
            }
        }

        return distances;
    }
};

// PointCloudProcessor class to handle point cloud data and A* planning
class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor() : Node("pointcloud_processor"){
        // publisher_ = this->create_publisher<std_msgs::msg::Bool>("flag_topic", 10);
        // Subscribe to the pointcloud and odometry topics
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", 10,
            std::bind(&PointCloudProcessor::odomCallback, this, std::placeholders::_1)
        );

        state_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "state_", 10, std::bind(&PointCloudProcessor::state_callback, this, std::placeholders::_1));

        key_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "keypress_status", 10, std::bind(&PointCloudProcessor::keypressed_callback, this, std::placeholders::_1));

        odom_reset_pose_client_ = this->create_client<rtabmap_msgs::srv::ResetPose>("reset_odom_to_pose");


        // Initialize parameters

        global_grid_rows_ = 200;
        global_grid_cols_ = 200;

        global_occupancy_grid_ = cv::Mat::zeros(global_grid_rows_, global_grid_cols_, CV_8UC1);

        // Set the entire 81st column to 1
        global_occupancy_grid_.col(70).setTo(1);
        global_occupancy_grid_.col(130).setTo(1);

        x_origin_ = global_grid_rows_ / 2;
        y_origin_ = global_grid_cols_ / 2;
        

        std::string image_path = "/home/redha/colcon_ws/src/btn/tasks/emptyseats.png";
        RCLCPP_INFO(this->get_logger(), "%s", image_path.c_str());
        image = cv::imread(image_path.c_str(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image." << std::endl;
            return;
        }
        cv::Mat rotated_image;
        cv::rotate(image, rotated_image, cv::ROTATE_90_COUNTERCLOCKWISE);

        cv::Mat resized_image;
        // cv::resize(rotated_image, resized_image, cv::Size(static_cast<int>(295), static_cast<int>(image.rows * 0.22)), 0, 0, cv::INTER_LINEAR);
        cv::resize(rotated_image, resized_image, cv::Size(60, 100), 0, 0, cv::INTER_LINEAR);

        int original_width = resized_image.cols;
        int original_height = resized_image.rows;
        // Target dimensions
        int target_size = 200;

        // Calculate the top, bottom, left, and right padding to make the image 1000x1000
        int top = (target_size - original_height);
        int bottom = 0;
        int left = (target_size - original_width) / 2;
        int right = target_size - original_width - left;

        // Pad the image to make it 1000x1000 with a black border
        cv::copyMakeBorder(resized_image, padded_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Set goals
        goals.push_back(std::make_tuple(199, 90, "front_seats"));
        goals.push_back(std::make_tuple(130, 119, "table_1"));
        // goals.push_back(std::make_tuple(124, 119, "table"));
        goals.push_back(std::make_tuple(119, 90, "behind_seats"));
        // goals.push_back(std::make_tuple(119, 112, "table_left"));
        goals.push_back(std::make_tuple(100, 100, "end_point"));
    }
    void state_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        detection_state = msg->data;
        task_audio_feedback = true;
    }

    void keypressed_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (!msg->data){
            key_press_counter++;
            if (key_press_counter == 2) {
                resetOdomWithPose(4.0, -0.65, 0.0, 0.0, 0.0, 1.57);
            }
        }
    }
    // Function to calculate the angle between the fitted line and the x-axis
    double fitLineAndGetAngle(const std::vector<std::pair<int, int>>& path) {
        // Extract 10 points (from index 5 to 15)
        std::vector<cv::Point2f> points;
        for (int i = 0; i < std::min(static_cast<size_t>(10), path.size()) && i < path.size(); ++i) {
            points.push_back(cv::Point2f(static_cast<float>(path[i].first), static_cast<float>(path[i].second)));
        }

        // Convert the points into a format suitable for cv::fitLine
        cv::Mat pointsMat(points.size(), 1, CV_32FC2, points.data());

        if (points.size() < 2){
            return 90.0;
        }
        // Fit a line to the points
        cv::Vec4f line;
        cv::fitLine(pointsMat, line, cv::DIST_L2, 0, 0.01, 0.01);

        // Extract line parameters: vx, vy, x0, y0
        float vx = line[0];  // Direction vector's x-component
        float vy = line[1];  // Direction vector's y-component
        float x0 = line[2];  // A point on the line (x-coordinate)
        float y0 = line[3];  // A point on the line (y-coordinate)

        // Calculate the slope of the line
        float slope = vy / vx;

        // Define two points on the line (x0, y0 and another distant point)
        int x1 = 500;  // Example x valueangle_radians
        int y1 = static_cast<int>(y0 + ((x1 - x0) * slope));  // Calculate corresponding y value

        // Calculate the angle using atan2
        double delta_x = x1 - x0;

        double delta_y = y1 - y0;
        double angle_radians = std::atan2(delta_y, delta_x);

        // Convert radians to degrees
        double angle_degrees = angle_radians * (180.0 / CV_PI);

        return angle_degrees + 90.0;
    }

    void resetOdomWithPose(float x, float y, float z, float roll, float pitch, float yaw)
    {
        // Create a request for the ResetPose service
        auto request = std::make_shared<rtabmap_msgs::srv::ResetPose::Request>();

        // Set the pose values (x, y, z, roll, pitch, yaw)
        request->x = x;
        request->y = y;
        request->z = z;
        request->roll = roll;
        request->pitch = pitch;
        request->yaw = yaw;

        // Call the service asynchronously
        auto result = odom_reset_pose_client_->async_send_request(request);

        current_state = "table_2";

        // key_pressed = true;


        // // Wait for the service call result
        // if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result) == rclcpp::FutureReturnCode::SUCCESS) {
        //     key_pressed = true;
        //     RCLCPP_INFO(this->get_logger(), "Odometry reset successfully to pose [x: %f, y: %f, z: %f, roll: %f, pitch: %f, yaw: %f].", x, y, z, roll, pitch, yaw);
        // } else {
        //     RCLCPP_ERROR(this->get_logger(), "Failed to reset odometry to specified pose.");
        // }
    }

void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
    current_z_ = msg->pose.pose.position.z;
    // Extract the quaternion from odometry
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
    
    // Convert quaternion to roll, pitch, and yaw
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    current_yaw_ = yaw;

    // Create a transformation that only uses the yaw value
    tf2::Transform odom_transform_yaw;
    tf2::Quaternion yaw_quat;
    yaw_quat.setRPY(0, 0, yaw);  // Set only pitch, ignoring roll and yaw
    odom_transform_yaw.setRotation(yaw_quat);
    odom_transform_yaw.setOrigin(tf2::Vector3(0, 0, 0));  // Zero out the translation

    // Store the inverse of the pitch-only transformation for point cloud alignment
    odom_to_start_ = odom_transform_yaw.inverse();


    // ############# State process #############33

    if ((strcmp(current_state.c_str(), "front_seats") == 0 || strcmp(current_state.c_str(), "behind_seats") == 0) && !detection_state) {
        return;
    }

    // if ((strcmp(current_state.c_str(), "table_2") == 0) && !key_pressed) {
    //     return;
    // }
 
    cv::Mat padded_image_32;
    padded_image.convertTo(padded_image_32, CV_8UC1);

    // padded_image_32.rowRange(0, std::max(std::min(160 - static_cast<int>(current_x_ * 20), 160), 1)).setTo(cv::Scalar(0));
    // padded_image_32.rowRange(std::max(std::min(199 - static_cast<int>(current_x_ * 20), 198), 0), 199).setTo(cv::Scalar(0));
    // padded_image_32.at<float>(198 - static_cast<int>(current_x_ * 20), 100 - static_cast<int>(current_y_ * 20)) = 1.0;

    cv::Mat refMat;
    cv::threshold(padded_image_32, refMat, 0, 1, cv::THRESH_BINARY);

    cv::Mat result;
    cv::bitwise_or(refMat, global_occupancy_grid_, result);

    cv::Mat planner_map;
    result.convertTo(planner_map, CV_8UC1);

    // Visualize the global map with the path
    cv::Mat displayGlobalGrid;
    result.convertTo(displayGlobalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display
    
    cv::Mat rgbGrid;
    cv::cvtColor(displayGlobalGrid, rgbGrid, cv::COLOR_GRAY2BGR);
    rgbGrid.at<cv::Vec3b>(std::min((199 - static_cast<int>(current_x_ * 20)), 199), (100 - static_cast<int>(current_y_ * 20))) = cv::Vec3b(0, 255, 0); // current position

    
    std::pair<int, int> start = {std::min((199 - static_cast<int>(current_x_ * 20)), 199), (100 - static_cast<int>(current_y_ * 20))};
    std::pair<int, int> goal = {100, 100}; // Example goal
    // std::pair<int, int> unchanged_goal = {100, 100}; // Example goal

    goal = {std::min(199, std::get<0>(goals[0])), std::get<1>(goals[0])};


    if (std::sqrt(std::pow(start.first - goal.first, 2) + std::pow(start.second - goal.second, 2)) < 5.0) {
        current_state = std::get<2>(goals[0]);
        goals.erase(goals.begin());
        task_audio_feedback = true;
        detection_state = false;
        if (strcmp(current_state.c_str(), "front_seats") == 0 || strcmp(current_state.c_str(), "behind_seats") == 0) {
        // if (strcmp(current_state.c_str(), "front_seats") == 0) {
            // auto message = std_msgs::msg::Bool();
            // message.data = true;
            // publisher_->publish(message);
            task_audio_feedback = false;
            system("espeak \"Ready!\"");
            RCLCPP_INFO(this->get_logger(), "Ready!");
            return;
        } else {
            system("espeak \"Reached!\"");
            RCLCPP_INFO(this->get_logger(), "Reached!");
        }
    }

    if (goals.empty()) {
        system("espeak \"Task Passed!\"");
        RCLCPP_INFO(this->get_logger(), "Task Passed!");
        exit(0);
    }

    if ((strcmp(current_state.c_str(), "start_point") == 0 || strcmp(current_state.c_str(), "behind_seats") == 0) && task_audio_feedback) {
        task_audio_feedback = false;
        system("espeak \"Walk Left\"");
        RCLCPP_INFO(this->get_logger(), "Walk Left");
    } else if ((strcmp(current_state.c_str(), "front_seats") == 0) || (strcmp(current_state.c_str(), "table_2") == 0)) {
        AStarPlanner planner(planner_map, 8);

        std::vector<std::pair<int, int>> path = planner.plan(start, goal);
        
        if (frame_counter == 15) {
            angle_path = fitLineAndGetAngle(path);
            frame_counter = 0;
        }

        for (int i = 0; i < path.size(); ++i) {
            const auto& p = path[i];
            rgbGrid.at<cv::Vec3b>(p.first, p.second) = cv::Vec3b(0, 0, 255);  // Red for the path
        }

        double angle_pilot = current_yaw_ * (180.0 / M_PI) + 90;

        double x = angle_path - angle_pilot;

        RCLCPP_INFO(this->get_logger(), "%f", angle_path);


        bool audio_feedback = true;  // Set this to true to enable audio feedback
        
        if (x < -45.0) {
            if (state != 'r') {
                RCLCPP_INFO(this->get_logger(), "Right");
                if (audio_feedback) {
                    system("espeak \"right\"");
                }
                state = 'r';
            }
        } else if (x > 45.0) {
            if (state != 'l') {
                RCLCPP_INFO(this->get_logger(), "Left");
                if (audio_feedback) {
                    system("espeak \"left\"");
                }
                state = 'l';
            }
        } else{
            if (state != 'f') {
                if (std::abs(x) < 10.0){
                    RCLCPP_INFO(this->get_logger(), "Forward");
                    if (audio_feedback) {
                        system("espeak \"Forward\"");
                    }
                    state = 'f';
                }
            }
        }
        frame_counter++;
    }

    // ############# Visualizations ###################

    rgbGrid.at<cv::Vec3b>(goal.first, goal.second) = cv::Vec3b(0, 255, 255); // goal position
    rgbGrid.at<cv::Vec3b>(std::get<0>(goals[0]), std::get<1>(goals[0])) = cv::Vec3b(0, 255, 255); // goal position
    rgbGrid.at<cv::Vec3b>(std::get<0>(goals[1]), std::get<1>(goals[1])) = cv::Vec3b(0, 255, 255); // goal position
    rgbGrid.at<cv::Vec3b>(std::get<0>(goals[2]), std::get<1>(goals[2])) = cv::Vec3b(0, 255, 255); // goal position
    // if (align_grids) {
    //     rgbGrid.at<cv::Vec3b>(goal.first, goal.second) = cv::Vec3b(0, 255, 255); // goal position
    //     // rgbGrid.at<cv::Vec3b>(unchanged_goal.first, unchanged_goal.second) = cv::Vec3b(255, 0, 0); // goal position
    // } 

    // Resize the grid from 200x200 to 800x800 for better visualization
    cv::Mat enlargedGrid;
    cv::resize(rgbGrid, enlargedGrid, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);  // Resize using nearest neighbor interpolation
    
    // Display the enlarged RGB matrix with path visualization
    cv::imshow("Enlarged Confidence Matrix in RGB", enlargedGrid);
    if (cv::waitKey(1) == 27) return;  // Exit on ESC key
}


private:
    // Subscribers for the pointcloud and odometry
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr state_subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr key_subscription_;

    // Variable to store the odometry transformation
    tf2::Transform odom_to_start_;
    tf2::Transform rotate_yaw_;

    // Current robot pose
    double current_x_, current_y_, current_z_, current_yaw_;

    // Global occupancy grid
    cv::Mat global_occupancy_grid_;
    int global_grid_rows_, global_grid_cols_;
    int x_origin_, y_origin_;


    char state = 'none';

    double angle_path = 90;

    int frame_counter = 0;

    std::chrono::steady_clock::time_point start_time_;

    cv::Mat image;
    cv::Mat padded_image;
    // bool align_grids = false;
    cv::Point bestMatchLoc;
    int mapPointsCount = 400;

    std::vector<std::tuple<int, int, std::string>> goals;
    bool task_audio_feedback = true;
    bool task_action = false;
    bool detection_state = false;

    std::string current_state = "start_point";
    int key_press_counter = 0;
    rclcpp::Client<rtabmap_msgs::srv::ResetPose>::SharedPtr odom_reset_pose_client_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
