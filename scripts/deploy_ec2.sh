#!/bin/bash
set -euo pipefail

# Values are inserted by GitHub Actions before the script is sent through SSM.
AWS_REGION="__AWS_REGION__"
IMAGE_URI="__IMAGE_URI__"

CONTAINER_NAME="sf-crime-api"
CONTAINER_PORT="8000"
HOST_PORT="8000"

READINESS_URL="http://127.0.0.1:${HOST_PORT}/health/ready"
LIVENESS_URL="http://127.0.0.1:${HOST_PORT}/health/live"

ECR_REGISTRY="${IMAGE_URI%%/*}"

echo "========================================"
echo "SF Crime production deployment"
echo "Image: ${IMAGE_URI}"
echo "========================================"

DOCKER_CONFIG_DIR="$(
    mktemp \
        --directory \
        /tmp/sf-crime-docker-config.XXXXXX
)"

cleanup_docker_credentials() {
    rm -rf "$DOCKER_CONFIG_DIR"
}

trap cleanup_docker_credentials EXIT

echo "Authenticating Docker with Amazon ECR."
aws ecr get-login-password \
    --region "$AWS_REGION" \
    | docker \
        --config "$DOCKER_CONFIG_DIR" \
        login \
        --username AWS \
        --password-stdin "$ECR_REGISTRY"

echo "Pulling the immutable deployment image."
docker \
    --config "$DOCKER_CONFIG_DIR" \
    pull "$IMAGE_URI"

docker \
    --config "$DOCKER_CONFIG_DIR" \
    logout "$ECR_REGISTRY" \
    >/dev/null 2>&1 \
    || true

cleanup_docker_credentials
trap - EXIT

PREVIOUS_IMAGE="$(
    docker inspect \
        --format='{{.Config.Image}}' \
        "$CONTAINER_NAME" \
        2>/dev/null \
        || true
)"

if [ -n "$PREVIOUS_IMAGE" ]; then
    echo "Previous image: $PREVIOUS_IMAGE"
else
    echo "No previous production image was found."
fi

rollback() {
    echo "Deployment failed. Beginning rollback."

    docker rm \
        --force \
        "$CONTAINER_NAME" \
        2>/dev/null \
        || true

    if [ -z "$PREVIOUS_IMAGE" ]; then
        echo "Rollback is unavailable because no previous image was found."
        exit 1
    fi

    echo "Restoring previous image: $PREVIOUS_IMAGE"

    docker run \
        --detach \
        --restart unless-stopped \
        --name "$CONTAINER_NAME" \
        --publish "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
        "$PREVIOUS_IMAGE"

    rollback_ready=false

    for attempt in $(seq 1 90); do
        echo "Rollback readiness attempt ${attempt}/90"

        if curl \
            --connect-timeout 3 \
            --max-time 8 \
            --fail \
            --silent \
            --show-error \
            "$READINESS_URL" \
            > /tmp/sf-crime-rollback-readiness.json; then
            rollback_ready=true
            break
        fi

        sleep 2
    done

    if [ "$rollback_ready" != "true" ]; then
        echo "CRITICAL: rollback container did not become ready."
        docker logs --tail 250 "$CONTAINER_NAME" || true
        exit 1
    fi

    echo "Rollback succeeded."
    cat /tmp/sf-crime-rollback-readiness.json
    exit 1
}

echo "Stopping the existing production container."

docker stop \
    --time 20 \
    "$CONTAINER_NAME" \
    2>/dev/null \
    || true

docker rm \
    "$CONTAINER_NAME" \
    2>/dev/null \
    || true

echo "Starting the replacement container."

if ! docker run \
    --detach \
    --restart unless-stopped \
    --name "$CONTAINER_NAME" \
    --publish "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
    "$IMAGE_URI"; then
    rollback
fi

deployment_ready=false

for attempt in $(seq 1 90); do
    echo "Deployment readiness attempt ${attempt}/90"

    if curl \
        --connect-timeout 3 \
        --max-time 8 \
        --fail \
        --silent \
        --show-error \
        "$READINESS_URL" \
        > /tmp/sf-crime-deployment-readiness.json; then
        deployment_ready=true
        break
    fi

    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "The replacement container disappeared."
        break
    fi

    container_state="$(
        docker inspect \
            --format='{{.State.Status}}' \
            "$CONTAINER_NAME" \
            2>/dev/null \
            || true
    )"

    if [ "$container_state" = "exited" ] || [ "$container_state" = "dead" ]; then
        echo "The replacement container entered state: $container_state"
        break
    fi

    sleep 2
done

if [ "$deployment_ready" != "true" ]; then
    echo "The replacement image failed its readiness check."

    docker inspect \
        --format='STATUS={{.State.Status}} HEALTH={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} OOM={{.State.OOMKilled}} EXIT={{.State.ExitCode}}' \
        "$CONTAINER_NAME" \
        2>/dev/null \
        || true

    docker logs --tail 250 "$CONTAINER_NAME" || true

    rollback
fi

echo "Readiness response:"
cat /tmp/sf-crime-deployment-readiness.json

echo
echo "Running liveness verification."

curl \
    --connect-timeout 3 \
    --max-time 10 \
    --fail \
    --silent \
    --show-error \
    "$LIVENESS_URL" \
    > /tmp/sf-crime-deployment-liveness.json

cat /tmp/sf-crime-deployment-liveness.json

echo
echo "Confirming deployed image."

DEPLOYED_IMAGE="$(
    docker inspect \
        --format='{{.Config.Image}}' \
        "$CONTAINER_NAME"
)"

if [ "$DEPLOYED_IMAGE" != "$IMAGE_URI" ]; then
    echo "Unexpected deployed image: $DEPLOYED_IMAGE"
    rollback
fi

docker ps \
    --filter "name=^/${CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'

echo "Removing dangling Docker layers."
docker image prune --force

echo "Deployment completed successfully."