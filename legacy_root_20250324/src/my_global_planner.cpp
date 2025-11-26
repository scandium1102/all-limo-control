#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <base_local_planner/costmap_model.h>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>

using std::vector;
using std::priority_queue;

namespace my_planner_namespace {

struct Node { unsigned int index; float f; };

// 比較仿函數，用於優先佇列依f值小者優先
struct CompareNode {
    bool operator()(const Node& a, const Node& b) {
        return a.f > b.f;
    }
};

class MyGlobalPlanner : public nav_core::BaseGlobalPlanner {
public:
    MyGlobalPlanner() : costmap_(nullptr), initialized_(false) {}
    MyGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
        : costmap_(nullptr), initialized_(false) {
        initialize(name, costmap_ros);
    }

    void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override {
        if (!initialized_) {
            costmap_ros_ = costmap_ros;
            costmap_ = costmap_ros_->getCostmap();
            // 可在此讀取參數伺服器配置（如需要）
            initialized_ = true;
            ROS_INFO("MyGlobalPlanner initialized");
        }
    }

    bool makePlan(const geometry_msgs::PoseStamped& start,
                  const geometry_msgs::PoseStamped& goal,
                  std::vector<geometry_msgs::PoseStamped>& plan) override {
        plan.clear();
        if (!initialized_) {
            ROS_ERROR("MyGlobalPlanner has not been initialized");
            return false;
        }
        // 確保起點和終點座標使用全域地圖座標系
        std::string global_frame = costmap_ros_->getGlobalFrameID();
        if (start.header.frame_id != global_frame || goal.header.frame_id != global_frame) {
            ROS_ERROR("Start or goal pose is not in the global costmap frame");
            return false;
        }
        // 將世界座標轉為地圖格點索引
        unsigned int start_x, start_y, goal_x, goal_y;
        if (!worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y) ||
            !worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
            ROS_WARN("Start or goal is out of the map bounds");
            return false;
        }
        unsigned int start_index = start_y * costmap_->getSizeInCellsX() + start_x;
        unsigned int goal_index  = goal_y  * costmap_->getSizeInCellsX() + goal_x;
        if (start_index == goal_index) {
            // 起點即終點
            plan.push_back(start);
            return true;
        }
        // 初始化A*搜索資料結構
        unsigned int width  = costmap_->getSizeInCellsX();
        unsigned int height = costmap_->getSizeInCellsY();
        unsigned int total_cells = width * height;
        vector<float> g_cost(total_cells, std::numeric_limits<float>::infinity());
        vector<int>   came_from(total_cells, -1);
        vector<bool>  closed(total_cells, false);
        priority_queue<Node, vector<Node>, CompareNode> open_list;
        // 啟動節點
        g_cost[start_index] = 0;
        float h_start = heuristic(start_x, start_y, goal_x, goal_y);
        open_list.push({ start_index, h_start });
        // A*主迴圈
        const int DIRS = 8;
        int dir_dx[DIRS] = {1,-1, 0, 0, 1, 1,-1,-1};
        int dir_dy[DIRS] = {0, 0, 1,-1, 1,-1, 1,-1};
        float dir_cost[DIRS] = {1.0,1.0,1.0,1.0, 1.4142,1.4142,1.4142,1.4142};
        bool found_path = false;
        while (!open_list.empty()) {
            Node current = open_list.top();
            open_list.pop();
            if (closed[current.index]) continue;
            closed[current.index] = true;
            if (current.index == goal_index) {
                found_path = true;
                break;
            }
            unsigned int cur_x = current.index % width;
            unsigned int cur_y = current.index / width;
            // 檢查8鄰居
            for (int i = 0; i < DIRS; ++i) {
                int nx = cur_x + dir_dx[i];
                int ny = cur_y + dir_dy[i];
                if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
                unsigned int n_index = ny * width + nx;
                if (closed[n_index]) continue;
                unsigned char cost = costmap_->getCost(nx, ny);
                // 排除障礙物或未知區域
                if (cost >= costmap_2d::LETHAL_OBSTACLE || cost == costmap_2d::NO_INFORMATION) continue;
                // 計算臨時g成本
                float step = dir_cost[i] + (float)cost / 252.0;  // 包含額外代價
                float new_g = g_cost[current.index] + step;
                if (new_g < g_cost[n_index]) {
                    g_cost[n_index] = new_g;
                    came_from[n_index] = current.index;
                    float h = heuristic(nx, ny, goal_x, goal_y);
                    open_list.push({ n_index, new_g + h });
                }
            }
        }
        if (!found_path) {
            ROS_WARN("Global planner failed to find a path!");
            return false;
        }
        // 重建從起點到目標的路徑
        reconstructPath(start_index, goal_index, came_from, plan);
        // 用精確的起終點替換路徑兩端
        plan.front() = start;
        plan.back() = goal;
        return true;
    }

private:
    costmap_2d::Costmap2DROS* costmap_ros_;
    costmap_2d::Costmap2D* costmap_;
    bool initialized_;
    // 將世界坐標(wx, wy)轉為地圖索引(mx, my)
    bool worldToMap(double wx, double wy, unsigned int& mx, unsigned int& my) {
        if (!costmap_) return false;
        double origin_x = costmap_->getOriginX(), origin_y = costmap_->getOriginY();
        double resolution = costmap_->getResolution();
        if (wx < origin_x || wy < origin_y) return false;
        mx = (unsigned int)((wx - origin_x) / resolution);
        my = (unsigned int)((wy - origin_y) / resolution);
        if (mx < costmap_->getSizeInCellsX() && my < costmap_->getSizeInCellsY()) return true;
        return false;
    }
    // 歐幾里得距離啟發函式
    inline float heuristic(int x1, int y1, int x2, int y2) {
        float dx = x2 - x1, dy = y2 - y1;
        return std::sqrt(dx * dx + dy * dy);
    }
    // 回溯came_from得到路徑，並轉為世界座標Pose
    void reconstructPath(unsigned int start_index, unsigned int goal_index,
                         const vector<int>& came_from,
                         vector<geometry_msgs::PoseStamped>& plan) {
        vector<unsigned int> indices;
        unsigned int current = goal_index;
        indices.push_back(current);
        // 迴溯鏈直到起點
        while (current != start_index) {
            current = came_from[current];
            if (current == -1) {  // 理論上不會發生（保險起見）
                ROS_ERROR("Failed to reconstruct path: incomplete path data.");
                return;
            }
            indices.push_back(current);
        }
        std::reverse(indices.begin(), indices.end());
        // 轉換每個節點為PoseStamped
        ros::Time now = ros::Time::now();
        std::string global_frame = costmap_ros_->getGlobalFrameID();
        for (unsigned int idx : indices) {
            unsigned int x = idx % costmap_->getSizeInCellsX();
            unsigned int y = idx / costmap_->getSizeInCellsX();
            double wx = costmap_->getOriginX() + (x + 0.5) * costmap_->getResolution();
            double wy = costmap_->getOriginY() + (y + 0.5) * costmap_->getResolution();
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = now;
            pose.header.frame_id = global_frame;
            pose.pose.position.x = wx;
            pose.pose.position.y = wy;
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;
            plan.push_back(pose);
        }
    }
};  // class MyGlobalPlanner

}  // namespace my_planner_namespace

// 將插件註冊為 BaseGlobalPlanner
PLUGINLIB_EXPORT_CLASS(my_planner_namespace::MyGlobalPlanner, nav_core::BaseGlobalPlanner)
