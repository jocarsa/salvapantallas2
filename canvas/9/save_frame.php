<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $sessionId = $_POST['sessionId'];
    $counter = $_POST['counter'];
    $frameData = $_POST['frameData'];

    // Create session directory if it doesn't exist
    $sessionDir = 'sessions/' . $sessionId;
    if (!file_exists($sessionDir)) {
        mkdir($sessionDir, 0777, true);
    }

    // Decode the base64 image data and save as PNG
    $frameData = str_replace('data:image/png;base64,', '', $frameData);
    $frameData = base64_decode($frameData);
    $filePath = $sessionDir . '/render' . $counter . '.png';
    file_put_contents($filePath, $frameData);

    echo "Frame saved successfully.";
} else {
    echo "Invalid request method.";
}
?>

