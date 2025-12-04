#!/bin/bash
# Quick Setup & Launcher Script
# Simplifies getting started with parallel training and Docker

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_info "Docker not installed. Skipping Docker checks."
        return 1
    fi
    
    if ! docker ps &> /dev/null; then
        print_info "Docker daemon not running. Start Docker first."
        return 1
    fi
    
    print_success "Docker is available"
    return 0
}

# Check if required Python packages are installed
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check for Python packages
    python_packages=("gymnasium" "stable_baselines3" "habitat_sim" "numpy" "quaternion")
    
    for package in "${python_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package installed"
        else
            print_info "$package not found - will need to install"
        fi
    done
}

# Check if models exist
check_models() {
    print_header "Checking Trained Models"
    
    if [ -f "models/lowlevel_curriculum_250k.zip" ]; then
        print_success "Navigation model found: models/lowlevel_curriculum_250k.zip"
    else
        print_info "Navigation model not found - need to train first"
    fi
    
    if [ -f "models/fetch/fetch_nav_250k_final.zip" ]; then
        print_success "Fetch model found: models/fetch/fetch_nav_250k_final.zip"
    else
        print_info "Fetch model not found - need to train first"
    fi
}

# Test environments
test_environments() {
    print_header "Testing Environments"
    
    print_info "Testing Navigation Environment..."
    if python -c "from src.simple_navigation_env import SimpleNavigationEnv; env = SimpleNavigationEnv(); obs, _ = env.reset(); env.close(); print('OK')" 2>&1 | grep -q "OK"; then
        print_success "Navigation environment test passed"
    else
        print_info "Navigation environment test failed"
    fi
    
    print_info "Testing Fetch Environment..."
    if python -c "from src.arm.fetch_navigation_env import FetchNavigationEnv; env = FetchNavigationEnv(); obs, _ = env.reset(); env.close(); print('OK')" 2>&1 | grep -q "OK"; then
        print_success "Fetch environment test passed"
    else
        print_info "Fetch environment test failed"
    fi
}

# Menu
show_menu() {
    print_header "HRL Training & Development Menu"
    
    echo "1) Check system dependencies"
    echo "2) Test environments"
    echo "3) Train navigation (CPU, 250k steps)"
    echo "4) Train fetch navigation (CPU, 250k steps)"
    echo "5) Train both in parallel (CPU)"
    echo "6) Test skill combination"
    echo "7) Docker - Build image"
    echo "8) Docker - Start training container"
    echo "9) Docker - Run training (interactive)"
    echo "10) View documentation"
    echo "11) Exit"
    echo ""
    read -p "Select option [1-11]: " choice
}

# Execute menu choice
execute_choice() {
    case $choice in
        1)
            check_dependencies
            ;;
        2)
            test_environments
            ;;
        3)
            print_header "Training Navigation Policy"
            python src/train_low_level_nav.py
            ;;
        4)
            print_header "Training Fetch Navigation Policy"
            python src/arm/train_fetch_nav.py
            ;;
        5)
            print_header "Starting Parallel Training"
            print_info "This will train both policies simultaneously"
            print_info "Logs: logs/parallel/"
            python launch_parallel_training.py --mode both
            ;;
        6)
            print_header "Testing Skill Combination"
            check_models
            python src/skill_executor.py
            ;;
        7)
            if check_docker; then
                print_header "Building Docker Image"
                docker build -t hrl-training:latest .
                print_success "Docker image built successfully"
            fi
            ;;
        8)
            if check_docker; then
                print_header "Starting Docker Container"
                docker-compose up -d hrl-training
                print_success "Container started"
                echo ""
                echo "To enter the container:"
                echo "  docker-compose exec hrl-training bash"
                echo ""
                echo "To view logs:"
                echo "  docker-compose logs -f hrl-training"
            fi
            ;;
        9)
            if check_docker; then
                print_header "Running Training in Docker"
                docker-compose exec hrl-training python launch_parallel_training.py --mode both
            fi
            ;;
        10)
            print_header "Documentation Files"
            echo ""
            echo "Main guides:"
            echo "  • README.md - Project overview"
            echo "  • IMPLEMENTATION_SUMMARY.md - This implementation summary"
            echo "  • .github/copilot-instructions.md - AI agent guide"
            echo "  • DOCKER.md - Docker setup and troubleshooting"
            echo ""
            read -p "Open which file? (readme/summary/copilot/docker/none): " doc_choice
            case $doc_choice in
                readme)
                    less README.md
                    ;;
                summary)
                    less IMPLEMENTATION_SUMMARY.md
                    ;;
                copilot)
                    less .github/copilot-instructions.md
                    ;;
                docker)
                    less DOCKER.md
                    ;;
                *)
                    echo "Skipped"
                    ;;
            esac
            ;;
        11)
            print_success "Exiting"
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
}

# Main loop
main() {
    print_header "HRL Training & Development Setup"
    
    # Print quick status
    print_info "Quick Status:"
    check_models
    
    # Show menu and loop
    while true; do
        show_menu
        execute_choice
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main
main
