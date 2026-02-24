<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $epochTime = $_POST['epochTime'];
    $frameNumber = $_POST['frameNumber'];

    // Create a directory with the epoch time if it doesn't exist
    $directory = "frames/{$epochTime}";
    if (!file_exists($directory)) {
        mkdir($directory, 0777, true);
    }

    // Path to save the frame
    $filePath = "{$directory}/frame_{$frameNumber}.png";

    // Save the frame
    if (move_uploaded_file($_FILES['frame']['tmp_name'], $filePath)) {
        echo "Frame saved successfully.";
    } else {
        echo "Failed to save frame.";
    }
} else {
    echo "Invalid request method.";
}
?>

