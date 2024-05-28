package com.pets.pets_service.Controllers;

import com.pets.pets_service.Exception.ResourceNotFoundException;
import com.pets.pets_service.Models.Veterinary;
import com.pets.pets_service.Repositories.VeterinaryRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("api/v1")
public class VeterinaryController {
    @Autowired
    private VeterinaryRepo veterinaryRepo;

    @GetMapping("/veterinaries")
    public List<Veterinary> getAllVeterinaries() {
        return veterinaryRepo.findAll();
    }
    @GetMapping("/veterinaries/{id}")
    public ResponseEntity<Veterinary> getVeterinaryById(@PathVariable(value = "id") Integer veterinaryId)
            throws ResourceNotFoundException {

        Veterinary veterinary = veterinaryRepo.findById(veterinaryId)
                .orElseThrow(() -> new ResourceNotFoundException("veterinary not found"));
        return ResponseEntity.ok().body(veterinary);
    }

    @GetMapping("/veterinaries/email/{email}")
        public ResponseEntity<Veterinary> getPPByEmail(@PathVariable(value = "email") String email)
            throws ResourceNotFoundException {

        Veterinary veterinary = veterinaryRepo.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("veterinary not found with email: " + email));
        return ResponseEntity.ok().body(veterinary);
    }

    @PostMapping("/veterinaries")
    public Veterinary createVeterinary(@Valid @RequestBody Veterinary veterinary) {
        return veterinaryRepo.save(veterinary);
    }

    @PutMapping("/veterinaries/{id}")
    public ResponseEntity<Veterinary> updateVeterinary(@PathVariable(value = "id") Integer veterinaryId,
                                               @Valid @RequestBody Veterinary veterinaryDetails) throws ResourceNotFoundException {
        Veterinary veterinary = veterinaryRepo.findById(veterinaryId)
                .orElseThrow(() -> new ResourceNotFoundException("veterinary not found"));
        veterinary.setFullName(veterinaryDetails.getFullName());
        veterinary.setId(veterinaryDetails.getId());
        veterinary.setPhone(veterinaryDetails.getPhone());
        veterinary.setOfficeAddress(veterinaryDetails.getOfficeAddress());
        veterinary.setEmail(veterinaryDetails.getEmail());
        veterinary.setSpeciality(veterinaryDetails.getSpeciality());

        final Veterinary updatedVeterinary = veterinaryRepo.save(veterinary);
        return ResponseEntity.ok(updatedVeterinary);
    }

    @DeleteMapping("/veterinaries/{id}")
    public Map<String, Boolean> deleteVeterinaryAccount(@PathVariable(value = "id") Integer veterinaryId)
            throws ResourceNotFoundException {
        Veterinary veterinary = veterinaryRepo.findById(veterinaryId)
                .orElseThrow(() -> new ResourceNotFoundException("veterinary not found"));
        veterinaryRepo.delete(veterinary);
        Map<String, Boolean> response = new HashMap<>();
        response.put("deleted", Boolean.TRUE);
        return response;
    }
}
