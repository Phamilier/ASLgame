using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowUser : MonoBehaviour
{
    public Canvas floatingCanvas;
    public GameObject floatingWindow;
    public Transform playerHead; // Reference to your camera/head transform

    void Start()
    {
        SetupFloatingWindow();
    }

    void SetupFloatingWindow()
    {
        // Position the window in front of the player
        Vector3 windowPosition = playerHead.position + playerHead.forward * 5f + playerHead.right * 3f;
        floatingWindow.transform.position = windowPosition;

        // Make it face the player
        floatingWindow.transform.LookAt(playerHead);
        floatingWindow.transform.Rotate(0, 180, 0); // Flip to face player correctly
    }
}
